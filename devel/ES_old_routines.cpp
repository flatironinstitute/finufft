// for ns = 2 to 15 :

const double ES_betaoverns[] = {2.208, 2.258, 2.284, 2.305, 2.309, 2.315, 2.323, 2.33, 2.335, 2.286, 2.342, 2.306, 2.345, 2.316 };

const double ES_esterrs[] = {0.0683, 0.00689, 0.000774, 8.01e-05, 8.16e-06, 8.35e-07, 8.5e-08, 8.62e-09, 8.55e-10, 1.47e-10, 8.03e-12, 1.43e-12, 8.62e-14, 3.68e-14 };


//FROM int setup_kernel(spread_opts &opts,double eps,double R)

  int n=0;    // choose nspread from list of available error estimates

  while (eps<fudgefac*ES_esterrs[n]) ++n;  // n is pointer to list of esterrs
  int ns = n+2;       // since the const list began at ns=2

  opts.ES_beta = 1.00 * ES_betaoverns[n] * ns;  // lookup table, obsolete

