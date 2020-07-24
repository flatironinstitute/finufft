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
            @F*)      # declarations: bash string replacement gets 4 combos...
                simp=${line//@F/ finufft}
                many=${simp//\(/many\(int ntr, }     # insert new 1st arg; esc (
                echo "::"
                echo ""
                echo "$simp"
                simp=${simp//finufft/finufftf}
                echo "${simp//double/single}"
                echo ""
                echo "$many"
                many=${many//finufft/finufftf}
                echo "${many//double/single}"
                # *** how fold it nicely to 80 chars somehow?
                ;;
            # rest are exact matches for whole line...
            @t)
                echo ""
                echo "  Computes via a fast algorithm, to precision eps, one or more transforms:"
                echo ""
                ;;
            @i)
                echo ""
                echo "  Inputs:"
                echo "    ntr    how many transforms (vectorized \"many\" functions only, else ntr=1)"
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
            @fe)        # flag and eps
                echo "    iflag  if >=0, uses +i in complex exponential, otherwise -i"
                echo "    eps    desired relative precision; smaller is slower. This can be chosen"
                echo "           from 1e-1 down to ~ 1e-14 (in double precision) or 1e-6 (in single)"
                ;;
            @o)        # opts and Outputs
                echo "    opts   pointer to options struct (see opts.rst), or NULL for defaults"
                echo ""
                echo "  Outputs:"
                ;;
            @r)        # ier and notes start
                echo "    return value  0: success, 1: success but warning, >1: error (see error.rst)"
                echo ""
                echo "  Notes:"
                echo "    * complex arrays interleave Re, Im values, and their size is stated with"
                echo "      dimensions ordered fastest to slowest."
                ;;
            @notes12)   # specific to type 1 & 2
                echo "    * Fourier frequency indices in each dimension i are the integers lying"
                echo "      in [-Ni/2, (Ni-1)/2]. See modeord in opts.rst for their ordering."
                echo "    * all $NU $CO must lie in [-3pi,3pi)."
                ;;
            *)
                # all else is passed through
                echo "$line"
                ;;
        esac
    done < $i > $o
    # (note sneaky use of pipes above, filters lines from $i, output to $o)
done

# debug note: to debug, best to echo "$stuff" 1>&2   so it goes to stderr.
