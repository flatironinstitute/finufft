#!/bin/bash
# Generate c?d.doc and cguru.doc from *.docsrc, from the docs/ directory.
# Contains some of their text content here too.
# Barnett 7/24/20.

# local expansions done before insertion
NU="nonuniform point"
NF="nonuniform frequency target"
CO=coordinates
LM="length M real array"
LN="length N real array"


# stage 1: flesh out *.docsrc (input i) to *.docexp (output o)...
for i in *.docsrc
do
    o=${i/.docsrc/.doc}
    #echo "$o"
    # create or overwrite output as 0-length file (not needed):  echo -n "" > $o
    while IFS= read -r line; do
        # define all tags (case-sens) and their actions here:
        case $line in
            *@F*)      # declarations: bash string replacement gets 4 combos...
                simp=${line//@F/finufft}
                many=${simp//\(/many\(int ntr, }     # insert new 1st arg; esc (
                echo "::"                            # parsed-literal not good
                echo ""
                echo "$simp"
                simp=${simp//finufft/finufftf}
                echo "${simp//double/float}"
                echo ""
                echo "$many"
                many=${many//finufft/finufftf}
                echo "${many//double/float}"
                ;;
            *@G*)      # guru declarations:
                line=${line//@G/finufft}
                echo "::"
                echo ""
                echo "$line"
                line=${line//finufft/finufftf}        # catches both instances
                echo "${line//double/float}"
                ;;
            # rest are exact matches for whole line...
            @t)
                echo ""
                echo "  Computes to precision eps, via a fast algorithm, one or more transforms of the form:"
                echo ""
                ;;
            @nt)
                echo "    ntr    how many transforms (only for vectorized \"many\" functions, else ntr=1)"
                ;;
            @mi)
                echo "    M      number of $NU sources"
                ;;
            @mo)
                echo "    M      number of $NU targets"
                ;;
            @n)
                echo "    N      number of $NF""s"
                ;;
            @xr)
                echo "    x      $NU""s in R ($LM)"
                ;;
            @x2r)
                echo "    x,y    $NU $CO in R^2 ($LM""s)"
                ;;
            @x3r)
                echo "    x,y,z  $NU $CO in R^3 ($LM""s)"
                ;;
            @x)
                echo "    x      $NU""s ($LM)"
                ;;
            @x2)
                echo "    x,y    $NU $CO ($LM""s)"
                ;;
            @x3)
                echo "    x,y,z  $NU $CO ($LM""s)"
                ;;
            @s)
                echo "    s      $NF""s in R ($LN)"
                ;;
            @s2)
                echo "    s,t    $NF $CO in R^2 ($LN""s)"
                ;;
            @s3)
                echo "    s,t,u  $NF $CO in R^3 ($LN""s)"
                ;;
            @ci)
                echo "    c      source strengths (size M*ntr complex array)"
                ;;
            @co)
                echo "    c      values at $NU targets (size M*ntr complex array)"
                ;;
            @f)
                echo "    iflag  if >=0, uses +i in complex exponential, otherwise -i"
                ;;
            @e)
                echo "    eps    desired relative precision; smaller is slower. This can be chosen"
                echo "           from 1e-1 down to ~ 1e-14 (in double precision) or 1e-6 (in single)"
                ;;
            @o)
                echo "    opts   pointer to options struct (see opts.rst), or NULL for defaults"
                ;;
            @r)
                echo "    return value  0: success, 1: success but warning, >1: error (see error.rst)"
                ;;
            @no)
                echo ""
                echo "  Notes:"
                echo "    * complex arrays interleave Re, Im values, and their size is stated with"
                echo "      dimensions ordered fastest to slowest."
                ;;
            @notes12)   # specific to type 1 & 2
                echo "    * Fourier frequency indices in each dimension i are the integers lying"
                echo "      in [-Ni/2, (Ni-1)/2]. See above, and modeord in opts.rst for possible orderings."
                ;;
            *)
                # all else is passed through
                echo "$line"
                ;;
        esac
    done < $i | fold -s -w 90 | sed -e '/::/! s/^/ /' > $o
    # (note sneaky use of pipes above, filters lines from $i, output to $o,
    # also wraps and adds initial space unless line has ::, to get .rst right)
    # sed -e '/../!s/^/ /'
done

# debug note: to debug, best to echo "$stuff" 1>&2   so it goes to stderr.
