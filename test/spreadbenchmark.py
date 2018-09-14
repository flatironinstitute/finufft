import commands
import re
import sys

# Test set
tests = []

#tests.append({"dim":1, "M":1e8, "N":1e8, "tol":1e-3})
#tests.append({"dim":1, "M":1e8, "N":1e8, "tol":1e-8})
#tests.append({"dim":1, "M":1e8, "N":1e8, "tol":1e-15})

#tests.append({"dim":3, "M":1e7, "N":1e7, "tol":1e-3})
#tests.append({"dim":3, "M":1e7, "N":1e7, "tol":1e-8})

tests.append({"dim":3, "M":1e5, "N":1e5, "tol":1e-15})
tests.append({"dim":3, "M":1e6, "N":1e6, "tol":1e-15})
tests.append({"dim":3, "M":1e7, "N":1e7, "tol":1e-15})


# Flags to send
flags = "1"

# Make flags (eg OMP=OFF)
makeflags = ""

# Command template
cmdtemplate = "test/spreadtestnd %(dim)d %(M)g %(N)g %(tol)g " + flags

# Regexps to use
spreadre = re.compile(r'dir=1.*\n.*\s(?P<speed>\d+[\d\.e+-]*) pts/s.*\n.*rel err.*\s(?P<err>\d+[\d\.e+-]*)')
interpre = re.compile(r'dir=2.*\n.*\s(?P<speed>\d+[\d\.e+-]*) pts/s.*\n.*rel err.*\s(?P<err>\d+[\d\.e+-]*)')

commit_A = sys.argv[1]
commit_B = sys.argv[2]

print "* Comparing commits '%s' and '%s'" % (commit_A, commit_B)

# Find out where we are, so we can come back
gitout = commands.getoutput("git status -b --porcelain")
pos = re.match("## (.+)\n", gitout).group(1)
if re.match("HEAD ", pos):
    home = commands.getoutput("git rev-parse HEAD")
    print "* Seems we are currently in detached head, will return to commit %s" % home
else:
    home = re.match("([^ \.]+)", pos).group(1)
    print "* Will return to branch %s" % home

# Command runner
def runCommand(cmd):
    print "> " + cmd
    status, output = commands.getstatusoutput(cmd)
    print output
    return output
    
# Test runner
def runTests():
    print "Running tests..."
    results = []
    for params in tests:
        cmd = cmdtemplate % params
        # Best of 3
        interp_speed = 0
        interp_err = 0
        spread_speed = 0
        spread_err = 0
        for i in [1,2,3]:
            output = runCommand(cmd).rstrip()
            ms = spreadre.search(output)
            mi = interpre.search(output)
            interp_speed = max(interp_speed, mi.group("speed"))
            interp_err = max(interp_err, mi.group("err"))
            spread_speed = max(spread_speed, ms.group("speed"))
            spread_err = max(spread_err, ms.group("err"))            
        results.append({"cmd":cmd,
                        "interp_speed":interp_speed,
                        "interp_err":interp_err,
                        "spread_speed":spread_speed,
                        "spread_err":spread_err })
    return results

# Code checkout machinery
def checkoutandmake(commit):
    if commit == "local":
        # Do nothin, just make it
        pass
    else:
        # Stash current source tree and check out commit
        print "Stashing changes and checking out %s..." % commit
        runCommand("git stash save")
        runCommand("git checkout %s" % commit)
    print "Making..."
    print commands.getoutput("make clean test/spreadtestnd " + makeflags)
    
def restore(commit, home):
    if commit == "local":
        # We just tested local, so do nothing
        pass
    else:
        # Return home and pop stash
        print "Checking out %s and popping stash..." % home
        runCommand("git checkout %s" % home)        
        runCommand("git stash pop")
    
# Run tests    
print "* Testing %s" % commit_A    
checkoutandmake(commit_A)
res_A = runTests()
restore(commit_A, home)

print "* Testing %s" % commit_B    
checkoutandmake(commit_B)
res_B = runTests()
restore(commit_B, home)

# Present results

format = "%-15s | %-15s | %-15s | %-15s | %-15s "
header = format % ("Commit", "Spread spd", "Interp spd", "Spread err", "Interp err")
print ""
print "Make flags: " + makeflags
print ""

for idx,params in enumerate(tests):
    print "=== Test: dim=%(dim)d, M=%(M)g, N=%(N)g, tol=%(tol)g" % params
    print header
    c = commit_A
    r = res_A[idx]
    print format % (c, r["spread_speed"]+" pts/s", r["interp_speed"]+" pts/s", r["spread_err"], r["interp_err"])
    c = commit_B
    r = res_B[idx]
    print format % (c, r["spread_speed"]+" pts/s", r["interp_speed"]+" pts/s", r["spread_err"], r["interp_err"])

    spread_speedup = float(res_B[idx]["spread_speed"]) / float(res_A[idx]["spread_speed"])*100 - 100
    spread_sign = "+" if spread_speedup > 0 else ""
    spread_speedup = spread_sign + "%.1f" % spread_speedup + "%"    
    interp_speedup = float(res_B[idx]["interp_speed"]) / float(res_A[idx]["interp_speed"])*100 - 100   
    interp_sign = "+" if interp_speedup > 0 else ""
    interp_speedup = interp_sign + "%.1f" % interp_speedup + "%"    
    print format % ("", spread_speedup, interp_speedup, "", "")

    print ""
