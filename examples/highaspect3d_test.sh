# test various poor aspect ratios in 3d
# Barnett 2/6/17

# fastest
./finufft3d_test 10 400 400 1e6 1e-12 2

# weird thing is this one is slowest even though z split is easy - RAM access?
./finufft3d_test 400 10 400 1e6 1e-12 2

# expect poor when split only along z:
./finufft3d_test 400 400 10 1e6 1e-12 2
