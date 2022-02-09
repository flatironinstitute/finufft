#include SCTL_INCLUDE(fft_wrapper.hpp)
#include SCTL_INCLUDE(legendre_rule.hpp)

#include <fstream>

// TODO: Replace work vectors with dynamic-arrays

namespace SCTL_NAMESPACE {

template <class Real> void SphericalHarmonics<Real>::Grid2SHC(const Vector<Real>& X, Long Nt, Long Np, Long p1, Vector<Real>& S, SHCArrange arrange){
  Long N = X.Dim() / (Np*Nt);
  assert(X.Dim() == N*Np*Nt);

  Vector<Real> B1(N*(p1+1)*(p1+1));
  Grid2SHC_(X, Nt, Np, p1, B1);
  SHCArrange0(B1, p1, S, arrange);
}

template <class Real> void SphericalHarmonics<Real>::SHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p0, Long Nt, Long Np, Vector<Real>* X, Vector<Real>* X_theta, Vector<Real>* X_phi){
  Vector<Real> B0;
  SHCArrange1(S, arrange, p0, B0);
  SHC2Grid_(B0, p0, Nt, Np, X, X_phi, X_theta);
}

template <class Real> void SphericalHarmonics<Real>::SHCEval(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& theta_phi, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M;
    assert(B0.Dim() == dof * M);

    B1.ReInit(dof, M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(0) == dof);
  assert(B1.Dim(1) == M);

  Matrix<Real> SHBasis;
  SHBasisEval(p0, theta_phi, SHBasis);
  assert(SHBasis.Dim(1) == M);
  Long N = SHBasis.Dim(0);

  { // Set X
    if (X.Dim() != N*dof) X.ReInit(N * dof);
    for (Long k0 = 0; k0 < N; k0++) {
      for (Long k1 = 0; k1 < dof; k1++) {
        Real X_ = 0;
        for (Long i = 0; i < M; i++) X_ += B1[k1][i] * SHBasis[k0][i];
        X[k0 * dof + k1] = X_;
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::SHC2Pole(const Vector<Real>& S, SHCArrange arrange, Long p0, Vector<Real>& P){
  Vector<Real> QP[2];
  { // Set QP // TODO: store these weights
    Vector<Real> x(1), alp;
    const Real SQRT2PI = sqrt<Real>(4 * const_pi<Real>());
    for (Long i = 0; i < 2; i++) {
      x = (i ? const_pi<Real>() : 0);
      LegPoly_(alp, x, p0);
      QP[i].ReInit(p0 + 1, alp.begin());
      QP[i] *= SQRT2PI;
    }
  }

  Long M, N;
  { // Set M, N
    M = 0;
    if (arrange == SHCArrange::ALL) M = 2*(p0+1)*(p0+1);
    if (arrange == SHCArrange::ROW_MAJOR) M = (p0+1)*(p0+2);
    if (arrange == SHCArrange::COL_MAJOR_NONZERO) M = (p0+1)*(p0+1);
    if (M == 0) return;
    N = S.Dim() / M;
    assert(S.Dim() == N * M);
  }
  if(P.Dim() != N * 2) P.ReInit(N * 2);

  if (arrange == SHCArrange::ALL) {
    #pragma omp parallel
    { // Compute pole
      Integer tid = omp_get_thread_num();
      Integer omp_p = omp_get_num_threads();

      Long a = (tid + 0) * N / omp_p;
      Long b = (tid + 1) * N / omp_p;
      for (Long i = a; i < b; i++) {
        Real P_[2] = {0, 0};
        for (Long j = 0; j < p0 + 1; j++) {
          P_[0] += S[i*M + j*(p0+1)*2] * QP[0][j];
          P_[1] += S[i*M + j*(p0+1)*2] * QP[1][j];
        }
        P[2*i+0] = P_[0];
        P[2*i+1] = P_[1];
      }
    }
  }
  if (arrange == SHCArrange::ROW_MAJOR) {
    #pragma omp parallel
    { // Compute pole
      Integer tid = omp_get_thread_num();
      Integer omp_p = omp_get_num_threads();

      Long a = (tid + 0) * N / omp_p;
      Long b = (tid + 1) * N / omp_p;
      for (Long i = a; i < b; i++) {
        Long idx = 0;
        Real P_[2] = {0, 0};
        for (Long j = 0; j < p0 + 1; j++) {
          P_[0] += S[i*M+idx] * QP[0][j];
          P_[1] += S[i*M+idx] * QP[1][j];
          idx += 2*(j+1);
        }
        P[2*i+0] = P_[0];
        P[2*i+1] = P_[1];
      }
    }
  }
  if (arrange == SHCArrange::COL_MAJOR_NONZERO) {
    #pragma omp parallel
    { // Compute pole
      Integer tid = omp_get_thread_num();
      Integer omp_p = omp_get_num_threads();

      Long a = (tid + 0) * N / omp_p;
      Long b = (tid + 1) * N / omp_p;
      for (Long i = a; i < b; i++) {
        Real P_[2] = {0, 0};
        for (Long j = 0; j < p0 + 1; j++) {
          P_[0] += S[i*M+j] * QP[0][j];
          P_[1] += S[i*M+j] * QP[1][j];
        }
        P[2*i+0] = P_[0];
        P[2*i+1] = P_[1];
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::WriteVTK(const char* fname, const Vector<Real>* S, const Vector<Real>* v_ptr, SHCArrange arrange, Long p0, Long p1, Real period, const Comm& comm){
  typedef double VTKReal;

  Vector<Real> SS;
  if (S == nullptr) {
    Integer p = 2;
    Integer Ncoeff = (p + 1) * (p + 1);
    Vector<Real> SSS(COORD_DIM * Ncoeff), SSS_grid;
    SSS.SetZero();
    SSS[1+0*p+0*Ncoeff] = sqrt<Real>(2.0)/sqrt<Real>(3.0);
    SSS[1+1*p+1*Ncoeff] = 1/sqrt<Real>(3.0);
    SSS[1+2*p+2*Ncoeff] = 1/sqrt<Real>(3.0);
    SphericalHarmonics<Real>::SHC2Grid(SSS, SHCArrange::COL_MAJOR_NONZERO, p, p+1, 2*p+2, &SSS_grid);
    SphericalHarmonics<Real>::Grid2SHC(SSS_grid, p+1, 2*p+2, p0, SS, arrange);
    S = &SS;
  }

  Vector<Real> X, Xp, V, Vp;
  { // Upsample X
    const Vector<Real>& X0=*S;
    SphericalHarmonics<Real>::SHC2Grid(X0, arrange, p0, p1+1, 2*p1, &X);
    SphericalHarmonics<Real>::SHC2Pole(X0, arrange, p0, Xp);
  }
  if(v_ptr){ // Upsample V
    const Vector<Real>& X0=*v_ptr;
    SphericalHarmonics<Real>::SHC2Grid(X0, arrange, p0, p1+1, 2*p1, &V);
    SphericalHarmonics<Real>::SHC2Pole(X0, arrange, p0, Vp);
  }

  std::vector<VTKReal> point_coord;
  std::vector<VTKReal> point_value;
  std::vector<int32_t> poly_connect;
  std::vector<int32_t> poly_offset;
  { // Set point_coord, point_value, poly_connect
    Long N_ves = X.Dim()/(2*p1*(p1+1)*COORD_DIM); // Number of vesicles
    assert(Xp.Dim() == N_ves*2*COORD_DIM);
    for(Long k=0;k<N_ves;k++){ // Set point_coord
      Real C[COORD_DIM]={0,0,0};
      if(period>0){
        for(Integer l=0;l<COORD_DIM;l++) C[l]=0;
        for(Long i=0;i<p1+1;i++){
          for(Long j=0;j<2*p1;j++){
            for(Integer l=0;l<COORD_DIM;l++){
              C[l]+=X[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))];
            }
          }
        }
        for(Integer l=0;l<COORD_DIM;l++) C[l]+=Xp[0+2*(l+k*COORD_DIM)];
        for(Integer l=0;l<COORD_DIM;l++) C[l]+=Xp[1+2*(l+k*COORD_DIM)];
        for(Integer l=0;l<COORD_DIM;l++) C[l]/=2*p1*(p1+1)+2;
        for(Integer l=0;l<COORD_DIM;l++) C[l]=(round(C[l]/period))*period;
      }

      for(Long i=0;i<p1+1;i++){
        for(Long j=0;j<2*p1;j++){
          for(Integer l=0;l<COORD_DIM;l++){
            point_coord.push_back(X[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))]-C[l]);
          }
        }
      }
      for(Integer l=0;l<COORD_DIM;l++) point_coord.push_back(Xp[0+2*(l+k*COORD_DIM)]-C[l]);
      for(Integer l=0;l<COORD_DIM;l++) point_coord.push_back(Xp[1+2*(l+k*COORD_DIM)]-C[l]);
    }

    if(v_ptr) {
      Long data__dof = V.Dim() / (N_ves * 2*p1*(p1+1));
      for(Long k=0;k<N_ves;k++){ // Set point_value
        for(Long i=0;i<p1+1;i++){
          for(Long j=0;j<2*p1;j++){
            for(Long l=0;l<data__dof;l++){
              point_value.push_back(V[j+2*p1*(i+(p1+1)*(l+k*data__dof))]);
            }
          }
        }
        for(Long l=0;l<data__dof;l++) point_value.push_back(Vp[0+2*(l+k*data__dof)]);
        for(Long l=0;l<data__dof;l++) point_value.push_back(Vp[1+2*(l+k*data__dof)]);
      }
    }

    for(Long k=0;k<N_ves;k++){
      for(Long j=0;j<2*p1;j++){
        Long i0= 0;
        Long i1=p1;
        Long j0=((j+0)       );
        Long j1=((j+1)%(2*p1));

        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*(p1+1)+0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j1);
        poly_offset.push_back(poly_connect.size());

        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*(p1+1)+1);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j1);
        poly_offset.push_back(poly_connect.size());
      }
      for(Long i=0;i<p1;i++){
        for(Long j=0;j<2*p1;j++){
          Long i0=((i+0)       );
          Long i1=((i+1)       );
          Long j0=((j+0)       );
          Long j1=((j+1)%(2*p1));
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j0);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j0);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j1);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j1);
          poly_offset.push_back(poly_connect.size());
        }
      }
    }
  }

  Integer np = comm.Size();
  Integer myrank = comm.Rank();

  std::vector<VTKReal>& coord=point_coord;
  std::vector<VTKReal>& value=point_value;
  std::vector<int32_t>& connect=poly_connect;
  std::vector<int32_t>& offset=poly_offset;

  Long pt_cnt=coord.size()/COORD_DIM;
  Long poly_cnt=poly_offset.size();

  // Open file for writing.
  std::stringstream vtufname;
  vtufname<<fname<<"_"<<std::setfill('0')<<std::setw(6)<<myrank<<".vtp";
  std::ofstream vtufile;
  vtufile.open(vtufname.str().c_str());
  if(vtufile.fail()) return;

  bool isLittleEndian;
  { // Set isLittleEndian
    uint16_t number = 0x1;
    uint8_t *numPtr = (uint8_t*)&number;
    isLittleEndian=(numPtr[0] == 1);
  }

  // Proceed to write to file.
  Long data_size=0;
  vtufile<<"<?xml version=\"1.0\"?>\n";
  if(isLittleEndian) vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  else               vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  //===========================================================================
  vtufile<<"  <PolyData>\n";
  vtufile<<"    <Piece NumberOfPoints=\""<<pt_cnt<<"\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""<<poly_cnt<<"\">\n";

  //---------------------------------------------------------------------------
  vtufile<<"      <Points>\n";
  vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+coord.size()*sizeof(VTKReal);
  vtufile<<"      </Points>\n";
  //---------------------------------------------------------------------------
  if(value.size()){ // value
    vtufile<<"      <PointData>\n";
    vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+value.size()*sizeof(VTKReal);
    vtufile<<"      </PointData>\n";
  }
  //---------------------------------------------------------------------------
  vtufile<<"      <Polys>\n";
  vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+connect.size()*sizeof(int32_t);
  vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+offset.size() *sizeof(int32_t);
  vtufile<<"      </Polys>\n";
  //---------------------------------------------------------------------------

  vtufile<<"    </Piece>\n";
  vtufile<<"  </PolyData>\n";
  //===========================================================================
  vtufile<<"  <AppendedData encoding=\"raw\">\n";
  vtufile<<"    _";

  int32_t block_size;
  block_size=coord.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&coord  [0], coord.size()*sizeof(VTKReal));
  if(value.size()){ // value
    block_size=value.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value  [0], value.size()*sizeof(VTKReal));
  }
  block_size=connect.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&connect[0], connect.size()*sizeof(int32_t));
  block_size=offset .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&offset [0], offset .size()*sizeof(int32_t));

  vtufile<<"\n";
  vtufile<<"  </AppendedData>\n";
  //===========================================================================
  vtufile<<"</VTKFile>\n";
  vtufile.close();


  if(myrank) return;
  std::stringstream pvtufname;
  pvtufname<<fname<<".pvtp";
  std::ofstream pvtufile;
  pvtufile.open(pvtufname.str().c_str());
  if(pvtufile.fail()) return;
  pvtufile<<"<?xml version=\"1.0\"?>\n";
  pvtufile<<"<VTKFile type=\"PPolyData\">\n";
  pvtufile<<"  <PPolyData GhostLevel=\"0\">\n";
  pvtufile<<"      <PPoints>\n";
  pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\"/>\n";
  pvtufile<<"      </PPoints>\n";
  if(value.size()){ // value
    pvtufile<<"      <PPointData>\n";
    pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\"/>\n";
    pvtufile<<"      </PPointData>\n";
  }
  {
    // Extract filename from path.
    std::stringstream vtupath;
    vtupath<<'/'<<fname;
    std::string pathname = vtupath.str();
    auto found = pathname.find_last_of("/\\");
    std::string fname_ = pathname.substr(found+1);
    for(Integer i=0;i<np;i++) pvtufile<<"      <Piece Source=\""<<fname_<<"_"<<std::setfill('0')<<std::setw(6)<<i<<".vtp\"/>\n";
  }
  pvtufile<<"  </PPolyData>\n";
  pvtufile<<"</VTKFile>\n";
  pvtufile.close();
}


template <class Real> void SphericalHarmonics<Real>::Grid2VecSHC(const Vector<Real>& X, Long Nt, Long Np, Long p0, Vector<Real>& S, SHCArrange arrange) {
  Long N = X.Dim() / (Np*Nt);
  assert(X.Dim() == N*Np*Nt);
  assert(N % COORD_DIM == 0);

  Vector<Real> B0(N*Nt*Np);
  { // Set B0
    Vector<Real> sin_phi(Np), cos_phi(Np);
    for (Long i = 0; i < Np; i++) {
      sin_phi[i] = sin(2 * const_pi<Real>() * i / Np);
      cos_phi[i] = cos(2 * const_pi<Real>() * i / Np);
    }
    const auto& Y = LegendreNodes(Nt - 1);
    assert(Y.Dim() == Nt);
    Long Ngrid = Nt * Np;
    for (Long k = 0; k < N; k+=COORD_DIM) {
      for (Long i = 0; i < Nt; i++) {
        Real sin_theta = sqrt<Real>(1 - Y[i]*Y[i]);
        Real cos_theta = Y[i];
        Real csc_theta = 1 / sin_theta;
        const auto X_ = X.begin() + (k*Nt+i)*Np;
        auto B0_ = B0.begin() + (k*Nt+i)*Np;
        for (Long j = 0; j < Np; j++) {
          StaticArray<Real,3> in;
          in[0] = X_[0*Ngrid+j];
          in[1] = X_[1*Ngrid+j];
          in[2] = X_[2*Ngrid+j];

          StaticArray<Real,9> Q;
          { // Set Q
            Q[0] = sin_theta*cos_phi[j]; Q[1] = sin_theta*sin_phi[j]; Q[2] = cos_theta;
            Q[3] = cos_theta*cos_phi[j]; Q[4] = cos_theta*sin_phi[j]; Q[5] =-sin_theta;
            Q[6] =          -sin_phi[j]; Q[7] =           cos_phi[j]; Q[8] =         0;
          }
          B0_[0*Ngrid+j] = ( Q[0] * in[0] + Q[1] * in[1] + Q[2] * in[2] );
          B0_[1*Ngrid+j] = ( Q[3] * in[0] + Q[4] * in[1] + Q[5] * in[2] ) * csc_theta;
          B0_[2*Ngrid+j] = ( Q[6] * in[0] + Q[7] * in[1] + Q[8] * in[2] ) * csc_theta;
        }
      }
    }
  }

  Long p_ = p0 + 1;
  Long M0 = (p0+1)*(p0+1);
  Long M_ = (p_+1)*(p_+1);
  Vector<Real> B1(N*M_);
  Grid2SHC_(B0, Nt, Np, p_, B1);

  Vector<Real> B2(N*M0);
  const Complex<Real> imag(0,1);
  for (Long i=0; i<N; i+=COORD_DIM) {
    for (Long m=0; m<=p0; m++) {
      for (Long n=m; n<=p0; n++) {
        auto read_coeff = [&](const Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          Complex<Real> c;
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            c.real = coeff[idx_real];
            if (m) c.imag = coeff[idx_imag];
          }
          return c;
        };
        auto write_coeff = [&](Complex<Real> c, Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            coeff[idx_real] = c.real;
            if (m) coeff[idx_imag] = c.imag;
          }
        };

        auto gr = [&](Long n, Long m) { return read_coeff(B1, i+0, p_, n, m); };
        auto gt = [&](Long n, Long m) { return read_coeff(B1, i+1, p_, n, m); };
        auto gp = [&](Long n, Long m) { return read_coeff(B1, i+2, p_, n, m); };

        Complex<Real> phiY, phiG, phiX;
        { // (phiG, phiX) <-- (gt, gp)
          auto A = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>(n*n * ((n+1)*(n+1) - m*m) / (Real)((2*n+1)*(2*n+3))) : 0); };
          auto B = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>((n+1)*(n+1) * (n*n - m*m) / (Real)((2*n+1)*(2*n-1))) : 0); };
          phiY = gr(n,m);
          phiG = (gt(n+1,m)*A(n,m) - gt(n-1,m)*B(n,m) - imag*m*gp(n,m)) * (1/(Real)(std::max<Long>(n,1)*(n+1)));
          phiX = (gp(n+1,m)*A(n,m) - gp(n-1,m)*B(n,m) + imag*m*gt(n,m)) * (1/(Real)(std::max<Long>(n,1)*(n+1)));
        }

        auto phiV = (phiG * (n + 0) - phiY) * (1/(Real)(2*n + 1));
        auto phiW = (phiG * (n + 1) + phiY) * (1/(Real)(2*n + 1));

        if (n==0) {
          phiW = 0;
          phiX = 0;
        }
        write_coeff(phiV, B2, i+0, p0, n, m);
        write_coeff(phiW, B2, i+1, p0, n, m);
        write_coeff(phiX, B2, i+2, p0, n, m);
      }
    }
  }

  SHCArrange0(B2, p0, S, arrange);
}

template <class Real> void SphericalHarmonics<Real>::VecSHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p0, Long Nt, Long Np, Vector<Real>& X) {
  Vector<Real> B0;
  SHCArrange1(S, arrange, p0, B0);

  Long p_ = p0 + 1;
  Long M0 = (p0+1)*(p0+1);
  Long M_ = (p_+1)*(p_+1);
  Long N = B0.Dim() / M0;
  assert(B0.Dim() == N*M0);
  assert(N % COORD_DIM == 0);

  Vector<Real> B1(N*M_);
  const Complex<Real> imag(0,1);
  for (Long i=0; i<N; i+=COORD_DIM) {
    for (Long m=0; m<=p_; m++) {
      for (Long n=m; n<=p_; n++) {
        auto read_coeff = [&](const Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          Complex<Real> c;
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            c.real = coeff[idx_real];
            if (m) c.imag = coeff[idx_imag];
          }
          return c;
        };
        auto write_coeff = [&](Complex<Real> c, Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            coeff[idx_real] = c.real;
            if (m) coeff[idx_imag] = c.imag;
          }
        };

        auto phiG = [&](Long n, Long m) {
          auto phiV = read_coeff(B0, i+0, p0, n, m);
          auto phiW = read_coeff(B0, i+1, p0, n, m);
          return phiV + phiW;
        };
        auto phiY = [&](Long n, Long m) {
          auto phiV = read_coeff(B0, i+0, p0, n, m);
          auto phiW = read_coeff(B0, i+1, p0, n, m);
          return phiW * n - phiV * (n + 1);
        };
        auto phiX = [&](Long n, Long m) {
          return read_coeff(B0, i+2, p0, n, m);
        };

        Complex<Real> gr, gt, gp;
        { // (gt, gp) <-- (phiG, phiX)
          auto A = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>(n*n * ((n+1)*(n+1) - m*m) / (Real)((2*n+1)*(2*n+3))) : 0); };
          auto B = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>((n+1)*(n+1) * (n*n - m*m) / (Real)((2*n+1)*(2*n-1))) : 0); };
          gr = phiY(n,m);
          gt = phiG(n-1,m)*A(n-1,m) - phiG(n+1,m)*B(n+1,m) - imag*m*phiX(n,m);
          gp = phiX(n-1,m)*A(n-1,m) - phiX(n+1,m)*B(n+1,m) + imag*m*phiG(n,m);
        }

        write_coeff(gr, B1, i+0, p_, n, m);
        write_coeff(gt, B1, i+1, p_, n, m);
        write_coeff(gp, B1, i+2, p_, n, m);
      }
    }
  }

  { // Set X
    SHC2Grid_(B1, p_, Nt, Np, &X);

    Vector<Real> sin_phi(Np), cos_phi(Np);
    for (Long i = 0; i < Np; i++) {
      sin_phi[i] = sin(2 * const_pi<Real>() * i / Np);
      cos_phi[i] = cos(2 * const_pi<Real>() * i / Np);
    }
    const auto& Y = LegendreNodes(Nt - 1);
    assert(Y.Dim() == Nt);
    Long Ngrid = Nt * Np;
    for (Long k = 0; k < N; k+=COORD_DIM) {
      for (Long i = 0; i < Nt; i++) {
        Real sin_theta = sqrt<Real>(1 - Y[i]*Y[i]);
        Real cos_theta = Y[i];
        Real csc_theta = 1 / sin_theta;
        auto X_ = X.begin() + (k*Nt+i)*Np;
        for (Long j = 0; j < Np; j++) {
          StaticArray<Real,3> in;
          in[0] = X_[0*Ngrid+j];
          in[1] = X_[1*Ngrid+j] * csc_theta;
          in[2] = X_[2*Ngrid+j] * csc_theta;

          StaticArray<Real,9> Q;
          { // Set Q
            Q[0] = sin_theta*cos_phi[j]; Q[1] = sin_theta*sin_phi[j]; Q[2] = cos_theta;
            Q[3] = cos_theta*cos_phi[j]; Q[4] = cos_theta*sin_phi[j]; Q[5] =-sin_theta;
            Q[6] =          -sin_phi[j]; Q[7] =           cos_phi[j]; Q[8] =         0;
          }
          X_[0*Ngrid+j] = ( Q[0] * in[0] + Q[3] * in[1] + Q[6] * in[2] );
          X_[1*Ngrid+j] = ( Q[1] * in[0] + Q[4] * in[1] + Q[7] * in[2] );
          X_[2*Ngrid+j] = ( Q[2] * in[0] + Q[5] * in[1] + Q[8] * in[2] );
        }
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::VecSHCEval(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& theta_phi, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Matrix<Real> SHBasis;
  VecSHBasisEval(p0, theta_phi, SHBasis);
  assert(SHBasis.Dim(1) == COORD_DIM * M);
  Long N = SHBasis.Dim(0) / COORD_DIM;

  { // Set X <-- Q * SHBasis * B1
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      StaticArray<Real,9> Q;
      { // Set Q
        Real cos_theta = cos(theta_phi[k0 * 2 + 0]);
        Real sin_theta = sin(theta_phi[k0 * 2 + 0]);
        Real cos_phi = cos(theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(theta_phi[k0 * 2 + 1]);
        Q[0] = sin_theta*cos_phi; Q[1] = sin_theta*sin_phi; Q[2] = cos_theta;
        Q[3] = cos_theta*cos_phi; Q[4] = cos_theta*sin_phi; Q[5] =-sin_theta;
        Q[6] =          -sin_phi; Q[7] =           cos_phi; Q[8] =         0;
      }
      for (Long k1 = 0; k1 < dof; k1++) { // Set X <-- Q * SHBasis * B1
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * SHBasis[k0 * COORD_DIM + j][i];
          }
        }
        X[(k0 * dof + k1) * COORD_DIM + 0] = Q[0] * in[0] + Q[3] * in[1] + Q[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = Q[1] * in[0] + Q[4] * in[1] + Q[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = Q[2] * in[0] + Q[5] * in[1] + Q[8] * in[2];
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::StokesEvalSL(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& coord, bool interior, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Long N, p_;
  Matrix<Real> SHBasis;
  Vector<Real> R, theta_phi;
  { // Set N, p_, R, SHBasis
    p_ = p0 + 1;
    Real M_ = (p_+1) * (p_+1);
    N = coord.Dim() / COORD_DIM;
    assert(coord.Dim() == N * COORD_DIM);

    R.ReInit(N);
    theta_phi.ReInit(2 * N);
    for (Long i = 0; i < N; i++) { // Set R, theta_phi
      ConstIterator<Real> x = coord.begin() + i * COORD_DIM;
      R[i] = sqrt<Real>(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      theta_phi[i * 2 + 0] = atan2(sqrt<Real>(x[0]*x[0] + x[1]*x[1]), x[2]);
      theta_phi[i * 2 + 1] = atan2(x[1], x[0]);
    }
    SHBasisEval(p_, theta_phi, SHBasis);
    assert(SHBasis.Dim(1) == M_);
    assert(SHBasis.Dim(0) == N);
    SCTL_UNUSED(M_);
  }

  Matrix<Real> StokesOp(N * COORD_DIM, COORD_DIM * M);
  for (Long i = 0; i < N; i++) { // Set StokesOp

    Real cos_theta, sin_theta, csc_theta, cos_phi, sin_phi;
    { // Set cos_theta, csc_theta, cos_phi, sin_phi
      cos_theta = cos(theta_phi[i * 2 + 0]);
      sin_theta = sin(theta_phi[i * 2 + 0]);
      csc_theta = 1 / sin_theta;
      cos_phi = cos(theta_phi[i * 2 + 1]);
      sin_phi = sin(theta_phi[i * 2 + 1]);
    }
    Complex<Real> imag(0,1), exp_phi(cos_phi, -sin_phi);

    const Real radius = R[i];
    Vector<Real> rpow;
    rpow.ReInit(p0 + 4);
    if (interior) {
      rpow[0] = 1 / radius;
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * radius;  // rpow[n] = r^(n-1)
    } else {
      rpow[0] = 1;
      const Real rinv = 1 / radius;
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * rinv;  // rpow[n] = r^(-n)
    }

    for (Long m = 0; m <= p0; m++) {
      for (Long n = m; n <= p0; n++) {
        auto write_coeff = [&](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.real;
            if (m) {
              idx += (p0+1-m);
              StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
            }
          }
        };

        Complex<Real> Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp;
        { // Set vector spherical harmonics
          auto Y = [&SHBasis,p_,i](Long n, Long m) {
            Complex<Real> c;
            if (0 <= m && m <= n && n <= p_) {
              Long idx = (2 * p_ - m + 2) * m - (m ? p_+1 : 0) + n;
              c.real = SHBasis[i][idx];
              if (m) {
                idx += (p_+1-m);
                c.imag = SHBasis[i][idx];
              }
            }
            return c;
          };
          auto Yt = [exp_phi, &Y, &R, i](Long n, Long m) {
            auto A = (0<=n && m<=n ? 0.5 * sqrt<Real>((n+m)*(n-m+1)) * (m-1==0?2.0:1.0) : 0);
            auto B = (0<=n && m<=n ? 0.5 * sqrt<Real>((n-m)*(n+m+1)) * (m+1==0?2.0:1.0) : 0);
            return (B / exp_phi * Y(n, m + 1) - A * exp_phi * Y(n, m - 1)) / R[i];
          };
          Complex<Real> Y_1 = Y(n + 0, m);
          Complex<Real> Y_1t = Yt(n + 0, m);

          Complex<Real> Ycsc_1 = Y_1 * csc_theta;
          if (fabs(sin_theta) == 0) {
            auto Y_csc0 = [exp_phi, cos_theta](Long n, Long m) {
              if (m == 1) return -sqrt<Real>((2*n+1)*n*(n+1)) * ((n%2==0) && (cos_theta<0) ? -1 : 1) * exp_phi;
              return Complex<Real>(0, 0);
            };
            Ycsc_1 = Y_csc0(n + 0, m);
          }

          auto SetVecSH = [&imag,n,m](Complex<Real>& Vr, Complex<Real>& Vt, Complex<Real>& Vp, Complex<Real>& Wr, Complex<Real>& Wt, Complex<Real>& Wp, Complex<Real>& Xr, Complex<Real>& Xt, Complex<Real>& Xp, const Complex<Real> C0, const Complex<Real> C1, const Complex<Real> C2) {
            Vr = C0 * (-n-1);
            Vt = C2;
            Vp = -imag * m * C1;

            Wr = C0 * n;
            Wt = C2;
            Wp = -imag * m * C1;

            Xr = 0;
            Xt = imag * m * C1;
            Xp = C2;
          };
          { // Set Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp
            auto C0 = Y_1;
            auto C1 = Ycsc_1;
            auto C2 = Y_1t * R[i];
            SetVecSH(Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp, C0, C1, C2);
          }
        }

        Complex<Real> SVr, SVt, SVp;
        Complex<Real> SWr, SWt, SWp;
        Complex<Real> SXr, SXt, SXp;
        if (interior) {
          Real a, b;
          a = n / (Real)((2 * n + 1) * (2 * n + 3)) * rpow[n + 2];
          b = -(n + 1) / (Real)(4 * n + 2) * (rpow[n] - rpow[n + 2]);
          SVr = a * Vr + b * Wr;
          SVt = a * Vt + b * Wt;
          SVp = a * Vp + b * Wp;

          a = (n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * rpow[n];
          SWr = a * Wr;
          SWt = a * Wt;
          SWp = a * Wp;

          a = 1 / (Real)(2 * n + 1) * rpow[n + 1];
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        } else {
          Real a, b;
          a = n / (Real)((2 * n + 1) * (2 * n + 3)) * rpow[n + 2];
          SVr = a * Vr;
          SVt = a * Vt;
          SVp = a * Vp;

          a = (n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * rpow[n];
          b = n / (Real)(4 * n + 2) * (rpow[n + 2] - rpow[n]);
          SWr = a * Wr + b * Vr;
          SWt = a * Wt + b * Vt;
          SWp = a * Wp + b * Vp;

          a = 1 / (Real)(2 * n + 1) * rpow[n + 1];
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        }

        write_coeff(SVr, n, m, 0, 0);
        write_coeff(SVt, n, m, 0, 1);
        write_coeff(SVp, n, m, 0, 2);

        write_coeff(SWr, n, m, 1, 0);
        write_coeff(SWt, n, m, 1, 1);
        write_coeff(SWp, n, m, 1, 2);

        write_coeff(SXr, n, m, 2, 0);
        write_coeff(SXt, n, m, 2, 1);
        write_coeff(SXp, n, m, 2, 2);
      }
    }
  }

  { // Set X <-- Q * StokesOp * B1
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      StaticArray<Real,9> Q;
      { // Set Q
        Real cos_theta = cos(theta_phi[k0 * 2 + 0]);
        Real sin_theta = sin(theta_phi[k0 * 2 + 0]);
        Real cos_phi = cos(theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(theta_phi[k0 * 2 + 1]);
        Q[0] = sin_theta*cos_phi; Q[1] = sin_theta*sin_phi; Q[2] = cos_theta;
        Q[3] = cos_theta*cos_phi; Q[4] = cos_theta*sin_phi; Q[5] =-sin_theta;
        Q[6] =          -sin_phi; Q[7] =           cos_phi; Q[8] =         0;
      }
      for (Long k1 = 0; k1 < dof; k1++) { // Set X <-- Q * StokesOp * B1
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * StokesOp[k0 * COORD_DIM + j][i];
          }
        }
        X[(k0 * dof + k1) * COORD_DIM + 0] = Q[0] * in[0] + Q[3] * in[1] + Q[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = Q[1] * in[0] + Q[4] * in[1] + Q[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = Q[2] * in[0] + Q[5] * in[1] + Q[8] * in[2];
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::StokesEvalDL(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& coord, bool interior, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Long N, p_;
  Matrix<Real> SHBasis;
  Vector<Real> R, theta_phi;
  { // Set N, p_, R, SHBasis
    p_ = p0 + 1;
    Real M_ = (p_+1) * (p_+1);
    N = coord.Dim() / COORD_DIM;
    assert(coord.Dim() == N * COORD_DIM);

    R.ReInit(N);
    theta_phi.ReInit(2 * N);
    for (Long i = 0; i < N; i++) { // Set R, theta_phi
      ConstIterator<Real> x = coord.begin() + i * COORD_DIM;
      R[i] = sqrt<Real>(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      theta_phi[i * 2 + 0] = atan2(sqrt<Real>(x[0]*x[0] + x[1]*x[1]), x[2]);
      theta_phi[i * 2 + 1] = atan2(x[1], x[0]);
    }
    SHBasisEval(p_, theta_phi, SHBasis);
    assert(SHBasis.Dim(1) == M_);
    assert(SHBasis.Dim(0) == N);
    SCTL_UNUSED(M_);
  }

  Matrix<Real> StokesOp(N * COORD_DIM, COORD_DIM * M);
  for (Long i = 0; i < N; i++) { // Set StokesOp

    Real cos_theta, sin_theta, csc_theta, cos_phi, sin_phi;
    { // Set cos_theta, csc_theta, cos_phi, sin_phi
      cos_theta = cos(theta_phi[i * 2 + 0]);
      sin_theta = sin(theta_phi[i * 2 + 0]);
      csc_theta = 1 / sin_theta;
      cos_phi = cos(theta_phi[i * 2 + 1]);
      sin_phi = sin(theta_phi[i * 2 + 1]);
    }
    Complex<Real> imag(0,1), exp_phi(cos_phi, -sin_phi);

    const Real radius = R[i];
    Vector<Real> rpow;
    rpow.ReInit(p0 + 4);
    if (interior) {
      rpow[0] = 1 / radius;
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * radius;  // rpow[n] = r^(n-1)
    } else {
      rpow[0] = 1;
      const Real rinv = 1 / radius;
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * rinv;  // rpow[n] = r^(-n)
    }

    for (Long m = 0; m <= p0; m++) {
      for (Long n = m; n <= p0; n++) {
        auto write_coeff = [&](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.real;
            if (m) {
              idx += (p0+1-m);
              StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
            }
          }
        };

        Complex<Real> Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp;
        { // Set vector spherical harmonics
          auto Y = [&SHBasis,p_,i](Long n, Long m) {
            Complex<Real> c;
            if (0 <= m && m <= n && n <= p_) {
              Long idx = (2 * p_ - m + 2) * m - (m ? p_+1 : 0) + n;
              c.real = SHBasis[i][idx];
              if (m) {
                idx += (p_+1-m);
                c.imag = SHBasis[i][idx];
              }
            }
            return c;
          };
          auto Yt = [exp_phi, &Y, &R, i](Long n, Long m) {
            auto A = (0<=n && m<=n ? 0.5 * sqrt<Real>((n+m)*(n-m+1)) * (m-1==0?2.0:1.0) : 0);
            auto B = (0<=n && m<=n ? 0.5 * sqrt<Real>((n-m)*(n+m+1)) * (m+1==0?2.0:1.0) : 0);
            return (B / exp_phi * Y(n, m + 1) - A * exp_phi * Y(n, m - 1)) / R[i];
          };
          Complex<Real> Y_1 = Y(n + 0, m);
          Complex<Real> Y_1t = Yt(n + 0, m);

          Complex<Real> Ycsc_1 = Y_1 * csc_theta;
          if (fabs(sin_theta) == 0) {
            auto Y_csc0 = [exp_phi, cos_theta](Long n, Long m) {
              if (m == 1) return -sqrt<Real>((2*n+1)*n*(n+1)) * ((n%2==0) && (cos_theta<0) ? -1 : 1) * exp_phi;
              return Complex<Real>(0, 0);
            };
            Ycsc_1 = Y_csc0(n + 0, m);
          }

          auto SetVecSH = [&imag,n,m](Complex<Real>& Vr, Complex<Real>& Vt, Complex<Real>& Vp, Complex<Real>& Wr, Complex<Real>& Wt, Complex<Real>& Wp, Complex<Real>& Xr, Complex<Real>& Xt, Complex<Real>& Xp, const Complex<Real> C0, const Complex<Real> C1, const Complex<Real> C2) {
            Vr = C0 * (-n-1);
            Vt = C2;
            Vp = -imag * m * C1;

            Wr = C0 * n;
            Wt = C2;
            Wp = -imag * m * C1;

            Xr = 0;
            Xt = imag * m * C1;
            Xp = C2;
          };
          { // Set Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp
            auto C0 = Y_1;
            auto C1 = Ycsc_1;
            auto C2 = Y_1t * R[i];
            SetVecSH(Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp, C0, C1, C2);
          }
        }

        Complex<Real> SVr, SVt, SVp;
        Complex<Real> SWr, SWt, SWp;
        Complex<Real> SXr, SXt, SXp;
        if (interior) {
          Real a, b;
          a = -2 * n * (n + 2) / (Real)((2 * n + 1) * (2 * n + 3)) * rpow[n + 2];  // pow<Real>(R[i], n+1);
          b = -(n + 1) * (n + 2) / (Real)(2 * n + 1) * (rpow[n + 2] - rpow[n]);    //(pow<Real>(R[i], n+1) - pow<Real>(R[i], n-1));
          SVr = a * Vr + b * Wr;
          SVt = a * Vt + b * Wt;
          SVp = a * Vp + b * Wp;

          a = -(2 * n * n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * rpow[n];  // pow<Real>(R[i], n-1);
          SWr = a * Wr;
          SWt = a * Wt;
          SWp = a * Wp;

          a = -(n + 2) / (Real)(2 * n + 1) * rpow[n + 1];  // pow<Real>(R[i], n);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        } else {
          Real a, b;
          a = (2 * n * n + 4 * n + 3) / (Real)((2 * n + 1) * (2 * n + 3)) * rpow[n + 2];  // pow<Real>(R[i], -n-2);
          SVr = a * Vr;
          SVt = a * Vt;
          SVp = a * Vp;

          a = 2 * (n + 1) * (n - 1) / (Real)((2 * n + 1) * (2 * n - 1)) * rpow[n];  // pow<Real>(R[i], -n);
          b = 2 * n * (n - 1) / (Real)(4 * n + 2) * (rpow[n + 2] - rpow[n]);        // (pow<Real>(R[i], -n-2) - pow<Real>(R[i], -n));
          SWr = a * Wr + b * Vr;
          SWt = a * Wt + b * Vt;
          SWp = a * Wp + b * Vp;

          a = (n - 1) / (Real)(2 * n + 1) * rpow[n + 1];  // pow<Real>(R[i], -n-1);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        }

        write_coeff(SVr, n, m, 0, 0);
        write_coeff(SVt, n, m, 0, 1);
        write_coeff(SVp, n, m, 0, 2);

        write_coeff(SWr, n, m, 1, 0);
        write_coeff(SWt, n, m, 1, 1);
        write_coeff(SWp, n, m, 1, 2);

        write_coeff(SXr, n, m, 2, 0);
        write_coeff(SXt, n, m, 2, 1);
        write_coeff(SXp, n, m, 2, 2);
      }
    }
  }

  { // Set X <-- Q * StokesOp * B1
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      StaticArray<Real,9> Q;
      { // Set Q
        Real cos_theta = cos(theta_phi[k0 * 2 + 0]);
        Real sin_theta = sin(theta_phi[k0 * 2 + 0]);
        Real cos_phi = cos(theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(theta_phi[k0 * 2 + 1]);
        Q[0] = sin_theta*cos_phi; Q[1] = sin_theta*sin_phi; Q[2] = cos_theta;
        Q[3] = cos_theta*cos_phi; Q[4] = cos_theta*sin_phi; Q[5] =-sin_theta;
        Q[6] =          -sin_phi; Q[7] =           cos_phi; Q[8] =         0;
      }
      for (Long k1 = 0; k1 < dof; k1++) { // Set X <-- Q * StokesOp * B1
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * StokesOp[k0 * COORD_DIM + j][i];
          }
        }
        X[(k0 * dof + k1) * COORD_DIM + 0] = Q[0] * in[0] + Q[3] * in[1] + Q[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = Q[1] * in[0] + Q[4] * in[1] + Q[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = Q[2] * in[0] + Q[5] * in[1] + Q[8] * in[2];
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::StokesEvalKL(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& coord, const Vector<Real>& norm, bool interior, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Long N, p_;
  Matrix<Real> SHBasis;
  Vector<Real> R, theta_phi;
  { // Set N, p_, R, SHBasis
    p_ = p0 + 2;
    Real M_ = (p_+1) * (p_+1);
    N = coord.Dim() / COORD_DIM;
    assert(coord.Dim() == N * COORD_DIM);

    R.ReInit(N);
    theta_phi.ReInit(2 * N);
    for (Long i = 0; i < N; i++) { // Set R, theta_phi
      ConstIterator<Real> x = coord.begin() + i * COORD_DIM;
      R[i] = sqrt<Real>(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      theta_phi[i * 2 + 0] = atan2(sqrt<Real>(x[0]*x[0] + x[1]*x[1]) + 1e-50, x[2]);
      theta_phi[i * 2 + 1] = atan2(x[1], x[0]);
    }
    SHBasisEval(p_, theta_phi, SHBasis);
    assert(SHBasis.Dim(1) == M_);
    assert(SHBasis.Dim(0) == N);
    SCTL_UNUSED(M_);
  }

  Matrix<Real> StokesOp(N * COORD_DIM, COORD_DIM * M);
  for (Long i = 0; i < N; i++) { // Set StokesOp

    Real cos_theta, sin_theta, csc_theta, cot_theta, cos_phi, sin_phi;
    { // Set cos_theta, sin_theta, cos_phi, sin_phi
      cos_theta = cos(theta_phi[i * 2 + 0]);
      sin_theta = sin(theta_phi[i * 2 + 0]);
      csc_theta = 1 / sin_theta;
      cot_theta = cos_theta * csc_theta;
      cos_phi = cos(theta_phi[i * 2 + 1]);
      sin_phi = sin(theta_phi[i * 2 + 1]);
    }
    Complex<Real> imag(0,1), exp_phi(cos_phi, -sin_phi);

    const Real radius = R[i];
    Vector<Real> rpow;
    rpow.ReInit(p0 + 4);
    if (interior) {
      rpow[0] = 1 / (radius * radius);
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * radius;  // rpow[n] = r^(n-2)
    } else {
      rpow[0] = 1;
      const Real rinv = 1 / radius;
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * rinv;  // rpow[n] = r^(-n)
    }

    StaticArray<Real, COORD_DIM> norm0;
    { // Set norm0 <-- Q^t * norm
      StaticArray<Real,9> Q;
      { // Set Q
        Q[0] = sin_theta*cos_phi; Q[1] = sin_theta*sin_phi; Q[2] = cos_theta;
        Q[3] = cos_theta*cos_phi; Q[4] = cos_theta*sin_phi; Q[5] =-sin_theta;
        Q[6] =          -sin_phi; Q[7] =           cos_phi; Q[8] =         0;
      }
      StaticArray<Real,COORD_DIM> in;
      in[0] = norm[i * COORD_DIM + 0];
      in[1] = norm[i * COORD_DIM + 1];
      in[2] = norm[i * COORD_DIM + 2];
      norm0[0] = Q[0] * in[0] + Q[1] * in[1] + Q[2] * in[2];
      norm0[1] = Q[3] * in[0] + Q[4] * in[1] + Q[5] * in[2];
      norm0[2] = Q[6] * in[0] + Q[7] * in[1] + Q[8] * in[2];
    }

    for (Long m = 0; m <= p0; m++) {
      for (Long n = m; n <= p0; n++) {
        auto write_coeff = [&](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.real;
            if (m) {
              idx += (p0+1-m);
              StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
            }
          }
        };

        Complex<Real> Ynm;
        Complex<Real> Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp;
        Complex<Real> Vr_t, Vt_t, Vp_t, Wr_t, Wt_t, Wp_t, Xr_t, Xt_t, Xp_t;
        Complex<Real> Vr_p, Vt_p, Vp_p, Wr_p, Wt_p, Wp_p, Xr_p, Xt_p, Xp_p;
        { // Set vector spherical harmonics
          auto Y = [&SHBasis,p_,i](Long n, Long m) {
            Complex<Real> c;
            if (0 <= m && m <= n && n <= p_) {
              Long idx = (2 * p_ - m + 2) * m - (m ? p_+1 : 0) + n;
              c.real = SHBasis[i][idx];
              if (m) {
                idx += (p_+1-m);
                c.imag = SHBasis[i][idx];
              }
            }
            return c;
          };
          auto Yt = [exp_phi, &Y, &R, i](Long n, Long m) {
            auto A = (0<=n && m<=n ? 0.5 * sqrt<Real>((n+m)*(n-m+1)) * (m-1==0?2.0:1.0) : 0);
            auto B = (0<=n && m<=n ? 0.5 * sqrt<Real>((n-m)*(n+m+1)) * (m+1==0?2.0:1.0) : 0);
            return (B / exp_phi * Y(n, m + 1) - A * exp_phi * Y(n, m - 1)) / R[i];
          };
          auto Yp = [&Y, &imag, &R, i, csc_theta](Long n, Long m) {
            return imag * m * Y(n, m) * csc_theta / R[i];
          };
          auto Ypt = [&Yt, &imag](Long n, Long m) {
            return imag * m * Yt(n, m);
          };
          auto Ytt = [sin_theta, exp_phi, &Yt, &R, i](Long n, Long m) {
            auto A = (0<=n && m<=n ? 0.5 * sqrt<Real>((n+m)*(n-m+1)) * (m-1==0?2.0:1.0) : 0);
            auto B = (0<=n && m<=n ? 0.5 * sqrt<Real>((n-m)*(n+m+1)) * (m+1==0?2.0:1.0) : 0);
            return (n==0 ? 0 : (B / exp_phi * Yt(n, m + 1) - A * exp_phi * Yt(n, m - 1)));
          };

          Complex<Real> Y_1 = Y(n + 0, m);

          Complex<Real> Y_0t = Yt(n - 1, m);
          Complex<Real> Y_1t = Yt(n + 0, m);
          Complex<Real> Y_2t = Yt(n + 1, m);

          //Complex<Real> Y_0p = Yp(n - 1, m);
          Complex<Real> Y_1p = Yp(n + 0, m);
          //Complex<Real> Y_2p = Yp(n + 1, m);

          auto Anm = (0<=n && m<=n && n<=p_ ? sqrt<Real>(n*n * ((n+1)*(n+1) - m*m) / (Real)((2*n+1)*(2*n+3))) : 0);
          auto Bnm = (0<=n && m<=n && n<=p_ ? sqrt<Real>((n+1)*(n+1) * (n*n - m*m) / (Real)((2*n+1)*(2*n-1))) : 0);

          auto SetVecSH = [&imag,n,m](Complex<Real>& Vr, Complex<Real>& Vt, Complex<Real>& Vp, Complex<Real>& Wr, Complex<Real>& Wt, Complex<Real>& Wp, Complex<Real>& Xr, Complex<Real>& Xt, Complex<Real>& Xp, const Complex<Real> C0, const Complex<Real> C1, const Complex<Real> C2) {
            Vr = C0 * (-n-1);
            Vt = C2;
            Vp = -imag * m * C1;

            Wr = C0 * n;
            Wt = C2;
            Wp = -imag * m * C1;

            Xr = 0;
            Xt = imag * m * C1;
            Xp = C2;
          };
          { // Set Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp
            auto C0 = Y_1;
            auto C1 = Y_1 * csc_theta;
            auto C2 = Yt(n,m) * R[i];
            SetVecSH(Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp, C0, C1, C2);
          }
          { // Set Vr_t, Vt_t, Vp_t, Wr_t, Wt_t, Wp_t, Xr_t, Xt_t, Xp_t
            auto C0 = Y_1t;
            auto C1 = (Y_1t - Y_1  * cot_theta / R[i]) * csc_theta;
            if (fabs(cos_theta) == 1 && m == 1) C1 = 0; ///////////// TODO

            auto C2 = Ytt(n,m);
            if (!m) C2 = (Anm * Y_2t - Bnm * Y_0t) * csc_theta - Y_1t * cot_theta; ///////////// TODO

            SetVecSH(Vr_t, Vt_t, Vp_t, Wr_t, Wt_t, Wp_t, Xr_t, Xt_t, Xp_t, C0, C1, C2);

            Vr_t += (-Vt) / R[i];
            Vt_t += ( Vr) / R[i];

            Wr_t += (-Wt) / R[i];
            Wt_t += ( Wr) / R[i];

            Xr_t += (-Xt) / R[i];
            Xt_t += ( Xr) / R[i];
          }
          { // Set Vr_p, Vt_p, Vp_p, Wr_p, Wt_p, Wp_p, Xr_p, Xt_p, Xp_p
            auto C0 = -Y_1p;
            auto C1 = -Y_1p * csc_theta;
            auto C2 = -Ypt(n, m) * csc_theta;
            //auto C2 = -(Anm * Y_2p - Bnm * Y_0p) * csc_theta;
            SetVecSH(Vr_p, Vt_p, Vp_p, Wr_p, Wt_p, Wp_p, Xr_p, Xt_p, Xp_p, C0, C1, C2);

            Vr_p += (-sin_theta * Vp                 ) * csc_theta / R[i];
            Vt_p += (-cos_theta * Vp                 ) * csc_theta / R[i];
            Vp_p += ( sin_theta * Vr + cos_theta * Vt) * csc_theta / R[i];

            Wr_p += (-sin_theta * Wp                 ) * csc_theta / R[i];
            Wt_p += (-cos_theta * Wp                 ) * csc_theta / R[i];
            Wp_p += ( sin_theta * Wr + cos_theta * Wt) * csc_theta / R[i];

            Xr_p += (-sin_theta * Xp                 ) * csc_theta / R[i];
            Xt_p += (-cos_theta * Xp                 ) * csc_theta / R[i];
            Xp_p += ( sin_theta * Xr + cos_theta * Xt) * csc_theta / R[i];

            if (fabs(cos_theta) == 1 && m == 1) {
              Vt_p = 0;
              Vp_p = 0;
              Wt_p = 0;
              Wp_p = 0;
              Xt_p = 0;
              Xp_p = 0;
            }
          }
          Ynm = Y_1;
        }

        if (fabs(cos_theta) == 1) {
          if (m!=0) Vr = 0;
          if (m!=1) Vt = 0;
          if (m!=1) Vp = 0;
          if (m!=0) Wr = 0;
          if (m!=1) Wt = 0;
          if (m!=1) Wp = 0;
          Xr = 0;
          if (m!=1) Xt = 0;
          if (m!=1) Xp = 0;

          if (m!=1        ) Vr_t = 0;
          if (m!=0 && m!=2) Vt_t = 0;
          if (m!=2        ) Vp_t = 0;
          if (m!=1        ) Wr_t = 0;
          if (m!=0 && m!=2) Wt_t = 0;
          if (m!=2        ) Wp_t = 0;
          if (m!=1        ) Xr_t = 0;
          if (m!=2        ) Xt_t = 0;
          if (m!=0 && m!=2) Xp_t = 0;

          if (m!=1        ) Vr_p = 0;
          if (m!=2        ) Vt_p = 0;
          if (m!=0 && m!=2) Vp_p = 0;
          if (m!=1        ) Wr_p = 0;
          if (m!=2        ) Wt_p = 0;
          if (m!=0 && m!=2) Wp_p = 0;
          if (m!=1        ) Xr_p = 0;
          if (m!=0 && m!=2) Xt_p = 0;
          if (m!=2        ) Xp_p = 0;
        }

        Complex<Real> PV, PW, PX;
        Complex<Real> SV[COORD_DIM][COORD_DIM];
        Complex<Real> SW[COORD_DIM][COORD_DIM];
        Complex<Real> SX[COORD_DIM][COORD_DIM];
        if (interior) {
          PV = (n + 1) * Ynm * rpow[n + 2];
          PW = 0;
          PX = 0;

          Real a, b;
          Real a_r, b_r;
          a = n / (Real)((2 * n + 1) * (2 * n + 3)) * rpow[n + 3];                           // pow<Real>(R[i], n+1);
          b = -(n + 1) / (Real)(4 * n + 2) * (rpow[n + 1] - rpow[n + 3]);                    // (pow<Real>(R[i], n-1) - pow<Real>(R[i], n+1));
          a_r = n / (Real)((2 * n + 1) * (2 * n + 3)) * (n + 1) * rpow[n + 2];               // pow<Real>(R[i], n);
          b_r = -(n + 1) / (Real)(4 * n + 2) * ((n - 1) * rpow[n] - (n + 1) * rpow[n + 2]);  // ((n-1) * pow<Real>(R[i], n-2) - (n+1) * pow<Real>(R[i], n));
          SV[0][0] = a_r * Vr + b_r * Wr;
          SV[1][0] = a_r * Vt + b_r * Wt;
          SV[2][0] = a_r * Vp + b_r * Wp;
          SV[0][1] = a * Vr_t + b * Wr_t;
          SV[1][1] = a * Vt_t + b * Wt_t;
          SV[2][1] = a * Vp_t + b * Wp_t;
          SV[0][2] = a * Vr_p + b * Wr_p;
          SV[1][2] = a * Vt_p + b * Wt_p;
          SV[2][2] = a * Vp_p + b * Wp_p;

          a = (n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * rpow[n + 1];          // pow<Real>(R[i], n-1);
          a_r = (n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * (n - 1) * rpow[n];  // pow<Real>(R[i], n-2);
          SW[0][0] = a_r * Wr;
          SW[1][0] = a_r * Wt;
          SW[2][0] = a_r * Wp;
          SW[0][1] = a * Wr_t;
          SW[1][1] = a * Wt_t;
          SW[2][1] = a * Wp_t;
          SW[0][2] = a * Wr_p;
          SW[1][2] = a * Wt_p;
          SW[2][2] = a * Wp_p;

          a = 1 / (Real)(2 * n + 1) * rpow[n + 2];        // pow<Real>(R[i], n);
          a_r = 1 / (Real)(2 * n + 1) * (n)*rpow[n + 1];  // pow<Real>(R[i], n-1);
          SX[0][0] = a_r * Xr;
          SX[1][0] = a_r * Xt;
          SX[2][0] = a_r * Xp;
          SX[0][1] = a * Xr_t;
          SX[1][1] = a * Xt_t;
          SX[2][1] = a * Xp_t;
          SX[0][2] = a * Xr_p;
          SX[1][2] = a * Xt_p;
          SX[2][2] = a * Xp_p;
        } else {
          PV = 0;
          PW = n * Ynm * rpow[n + 1];
          PX = 0;

          Real a, b;
          Real a_r, b_r;
          a = n / (Real)((2 * n + 1) * (2 * n + 3)) * rpow[n + 2];               // pow<Real>(R[i], -n-2);
          a_r = n / (Real)((2 * n + 1) * (2 * n + 3)) * (-n - 2) * rpow[n + 3];  // pow<Real>(R[i], -n-3);
          SV[0][0] = a_r * Vr;
          SV[1][0] = a_r * Vt;
          SV[2][0] = a_r * Vp;
          SV[0][1] = a * Vr_t;
          SV[1][1] = a * Vt_t;
          SV[2][1] = a * Vp_t;
          SV[0][2] = a * Vr_p;
          SV[1][2] = a * Vt_p;
          SV[2][2] = a * Vp_p;

          a = (n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * rpow[n];                    // pow<Real>(R[i], -n);
          b = n / (Real)(4 * n + 2) * (rpow[n + 2] - rpow[n]);                          //(pow<Real>(R[i], -n-2) - pow<Real>(R[i], -n));
          a_r = (n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * (-n) * rpow[n + 1];       // pow<Real>(R[i], -n-1);
          b_r = n / (Real)(4 * n + 2) * ((-n - 2) * rpow[n + 3] - (-n) * rpow[n + 1]);  // ((-n-2)*pow<Real>(R[i], -n-3) - (-n)*pow<Real>(R[i], -n-1));
          SW[0][0] = a_r * Wr + b_r * Vr;
          SW[1][0] = a_r * Wt + b_r * Vt;
          SW[2][0] = a_r * Wp + b_r * Vp;
          SW[0][1] = a * Wr_t + b * Vr_t;
          SW[1][1] = a * Wt_t + b * Vt_t;
          SW[2][1] = a * Wp_t + b * Vp_t;
          SW[0][2] = a * Wr_p + b * Vr_p;
          SW[1][2] = a * Wt_p + b * Vt_p;
          SW[2][2] = a * Wp_p + b * Vp_p;

          a = 1 / (Real)(2 * n + 1) * rpow[n + 1];               // pow<Real>(R[i], -n-1);
          a_r = 1 / (Real)(2 * n + 1) * (-n - 1) * rpow[n + 2];  // pow<Real>(R[i], -n-2);
          SX[0][0] = a_r * Xr;
          SX[1][0] = a_r * Xt;
          SX[2][0] = a_r * Xp;
          SX[0][1] = a * Xr_t;
          SX[1][1] = a * Xt_t;
          SX[2][1] = a * Xp_t;
          SX[0][2] = a * Xr_p;
          SX[1][2] = a * Xt_p;
          SX[2][2] = a * Xp_p;
        }

        Complex<Real> KV[COORD_DIM][COORD_DIM], KW[COORD_DIM][COORD_DIM], KX[COORD_DIM][COORD_DIM];
        KV[0][0] = SV[0][0] + SV[0][0] - PV;   KV[0][1] = SV[0][1] + SV[1][0]     ;   KV[0][2] = SV[0][2] + SV[2][0]     ;
        KV[1][0] = SV[1][0] + SV[0][1]     ;   KV[1][1] = SV[1][1] + SV[1][1] - PV;   KV[1][2] = SV[1][2] + SV[2][1]     ;
        KV[2][0] = SV[2][0] + SV[0][2]     ;   KV[2][1] = SV[2][1] + SV[1][2]     ;   KV[2][2] = SV[2][2] + SV[2][2] - PV;

        KW[0][0] = SW[0][0] + SW[0][0] - PW;   KW[0][1] = SW[0][1] + SW[1][0]     ;   KW[0][2] = SW[0][2] + SW[2][0]     ;
        KW[1][0] = SW[1][0] + SW[0][1]     ;   KW[1][1] = SW[1][1] + SW[1][1] - PW;   KW[1][2] = SW[1][2] + SW[2][1]     ;
        KW[2][0] = SW[2][0] + SW[0][2]     ;   KW[2][1] = SW[2][1] + SW[1][2]     ;   KW[2][2] = SW[2][2] + SW[2][2] - PW;

        KX[0][0] = SX[0][0] + SX[0][0] - PX;   KX[0][1] = SX[0][1] + SX[1][0]     ;   KX[0][2] = SX[0][2] + SX[2][0]     ;
        KX[1][0] = SX[1][0] + SX[0][1]     ;   KX[1][1] = SX[1][1] + SX[1][1] - PX;   KX[1][2] = SX[1][2] + SX[2][1]     ;
        KX[2][0] = SX[2][0] + SX[0][2]     ;   KX[2][1] = SX[2][1] + SX[1][2]     ;   KX[2][2] = SX[2][2] + SX[2][2] - PX;


        write_coeff(KV[0][0]*norm0[0] + KV[0][1]*norm0[1] + KV[0][2]*norm0[2], n, m, 0, 0);
        write_coeff(KV[1][0]*norm0[0] + KV[1][1]*norm0[1] + KV[1][2]*norm0[2], n, m, 0, 1);
        write_coeff(KV[2][0]*norm0[0] + KV[2][1]*norm0[1] + KV[2][2]*norm0[2], n, m, 0, 2);

        write_coeff(KW[0][0]*norm0[0] + KW[0][1]*norm0[1] + KW[0][2]*norm0[2], n, m, 1, 0);
        write_coeff(KW[1][0]*norm0[0] + KW[1][1]*norm0[1] + KW[1][2]*norm0[2], n, m, 1, 1);
        write_coeff(KW[2][0]*norm0[0] + KW[2][1]*norm0[1] + KW[2][2]*norm0[2], n, m, 1, 2);

        write_coeff(KX[0][0]*norm0[0] + KX[0][1]*norm0[1] + KX[0][2]*norm0[2], n, m, 2, 0);
        write_coeff(KX[1][0]*norm0[0] + KX[1][1]*norm0[1] + KX[1][2]*norm0[2], n, m, 2, 1);
        write_coeff(KX[2][0]*norm0[0] + KX[2][1]*norm0[1] + KX[2][2]*norm0[2], n, m, 2, 2);
      }
    }
  }

  { // Set X <-- Q * StokesOp * B1
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      StaticArray<Real,9> Q;
      { // Set Q
        Real cos_theta = cos(theta_phi[k0 * 2 + 0]);
        Real sin_theta = sin(theta_phi[k0 * 2 + 0]);
        Real cos_phi = cos(theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(theta_phi[k0 * 2 + 1]);
        Q[0] = sin_theta*cos_phi; Q[1] = sin_theta*sin_phi; Q[2] = cos_theta;
        Q[3] = cos_theta*cos_phi; Q[4] = cos_theta*sin_phi; Q[5] =-sin_theta;
        Q[6] =          -sin_phi; Q[7] =           cos_phi; Q[8] =         0;
      }
      for (Long k1 = 0; k1 < dof; k1++) { // Set X <-- Q * StokesOp * B1
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * StokesOp[k0 * COORD_DIM + j][i];
          }
        }
        X[(k0 * dof + k1) * COORD_DIM + 0] = Q[0] * in[0] + Q[3] * in[1] + Q[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = Q[1] * in[0] + Q[4] * in[1] + Q[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = Q[2] * in[0] + Q[5] * in[1] + Q[8] * in[2];
      }
    }
  }
}


template <class Real> void SphericalHarmonics<Real>::StokesEvalKSelf(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& coord, bool interior, Vector<Real>& X) {
 Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Long N, p_;
  Matrix<Real> SHBasis;
  Vector<Real> R, theta_phi;
  { // Set N, p_, R, SHBasis
    p_ = p0 + 1;
    Real M_ = (p_+1) * (p_+1);
    N = coord.Dim() / COORD_DIM;
    assert(coord.Dim() == N * COORD_DIM);

    R.ReInit(N);
    theta_phi.ReInit(2 * N);
    for (Long i = 0; i < N; i++) { // Set R, theta_phi
      ConstIterator<Real> x = coord.begin() + i * COORD_DIM;
      R[i] = sqrt<Real>(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      theta_phi[i * 2 + 0] = atan2(sqrt<Real>(x[0]*x[0] + x[1]*x[1]), x[2]);
      theta_phi[i * 2 + 1] = atan2(x[1], x[0]);
    }
    SHBasisEval(p_, theta_phi, SHBasis);
    assert(SHBasis.Dim(1) == M_);
    assert(SHBasis.Dim(0) == N);
    SCTL_UNUSED(M_);
  }

  Matrix<Real> StokesOp(N * COORD_DIM, COORD_DIM * M);
  for (Long i = 0; i < N; i++) { // Set StokesOp

    Real cos_theta, sin_theta, csc_theta, cos_phi, sin_phi;
    { // Set cos_theta, csc_theta, cos_phi, sin_phi
      cos_theta = cos(theta_phi[i * 2 + 0]);
      sin_theta = sin(theta_phi[i * 2 + 0]);
      csc_theta = 1 / sin_theta;
      cos_phi = cos(theta_phi[i * 2 + 1]);
      sin_phi = sin(theta_phi[i * 2 + 1]);
    }
    Complex<Real> imag(0,1), exp_phi(cos_phi, -sin_phi);

    const Real radius = R[i];
    Vector<Real> rpow;
    rpow.ReInit(p0 + 4);
    if (interior) {
      rpow[0] = 1 / (radius * radius);
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * radius;  // rpow[n] = r^(n-2)
    } else {
      rpow[0] = 1;
      const Real rinv = 1 / radius;
      for (Long ri = 1; ri < p0 + 4; ri++) rpow[ri] = rpow[ri - 1] * rinv;  // rpow[n] = r^(-n)
    }

    for (Long m = 0; m <= p0; m++) {
      for (Long n = m; n <= p0; n++) {
        auto write_coeff = [&](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.real;
            if (m) {
              idx += (p0+1-m);
              StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
            }
          }
        };

        Complex<Real> Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp;
        { // Set vector spherical harmonics
          auto Y = [&SHBasis,p_,i](Long n, Long m) {
            Complex<Real> c;
            if (0 <= m && m <= n && n <= p_) {
              Long idx = (2 * p_ - m + 2) * m - (m ? p_+1 : 0) + n;
              c.real = SHBasis[i][idx];
              if (m) {
                idx += (p_+1-m);
                c.imag = SHBasis[i][idx];
              }
            }
            return c;
          };
          auto Yt = [exp_phi, &Y, &R, i](Long n, Long m) {
            auto A = (0<=n && m<=n ? 0.5 * sqrt<Real>((n+m)*(n-m+1)) * (m-1==0?2.0:1.0) : 0);
            auto B = (0<=n && m<=n ? 0.5 * sqrt<Real>((n-m)*(n+m+1)) * (m+1==0?2.0:1.0) : 0);
            return (B / exp_phi * Y(n, m + 1) - A * exp_phi * Y(n, m - 1)) / R[i];
          };
          Complex<Real> Y_1 = Y(n + 0, m);
          Complex<Real> Y_1t = Yt(n + 0, m);

          Complex<Real> Ycsc_1 = Y_1 * csc_theta;
          if (fabs(sin_theta) == 0) {
            auto Y_csc0 = [exp_phi, cos_theta](Long n, Long m) {
              if (m == 1) return -sqrt<Real>((2*n+1)*n*(n+1)) * ((n%2==0) && (cos_theta<0) ? -1 : 1) * exp_phi;
              return Complex<Real>(0, 0);
            };
            Ycsc_1 = Y_csc0(n + 0, m);
          }

          auto SetVecSH = [&imag,n,m](Complex<Real>& Vr, Complex<Real>& Vt, Complex<Real>& Vp, Complex<Real>& Wr, Complex<Real>& Wt, Complex<Real>& Wp, Complex<Real>& Xr, Complex<Real>& Xt, Complex<Real>& Xp, const Complex<Real> C0, const Complex<Real> C1, const Complex<Real> C2) {
            Vr = C0 * (-n-1);
            Vt = C2;
            Vp = -imag * m * C1;

            Wr = C0 * n;
            Wt = C2;
            Wp = -imag * m * C1;

            Xr = 0;
            Xt = imag * m * C1;
            Xp = C2;
          };
          { // Set Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp
            auto C0 = Y_1;
            auto C1 = Ycsc_1;
            auto C2 = Y_1t * R[i];
            SetVecSH(Vr, Vt, Vp, Wr, Wt, Wp, Xr, Xt, Xp, C0, C1, C2);
          }
        }

        Complex<Real> SVr, SVt, SVp;
        Complex<Real> SWr, SWt, SWp;
        Complex<Real> SXr, SXt, SXp;
        if (interior) {
          Real a, b;
          a = ((2 * n * n + 4 * n + 3) / (Real)((2 * n + 1) * (2 * n + 3))) * rpow[n + 2];  // pow<Real>(R[i], n);
          b = ((n + 1) * (n - 1) / (Real)(2 * n + 1)) * (rpow[n + 2] - rpow[n]);            //(pow<Real>(R[i], n) - pow<Real>(R[i], n - 2));
          SVr = a * Vr + b * Wr;
          SVt = a * Vt + b * Wt;
          SVp = a * Vp + b * Wp;

          a = (2 * (n + 1) * (n - 1) / (Real)((2 * n + 1) * (2 * n - 1))) * rpow[n];  // * pow<Real>(R[i], n - 2);
          SWr = a * Wr;
          SWt = a * Wt;
          SWp = a * Wp;

          a = ((n - 1) / (Real)(2 * n + 1)) * rpow[n + 1];  // pow<Real>(R[i], n - 1);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        } else {
          Real a, b;
          a = -2 * n * (n + 2) / (Real)((2 * n + 1) * (2 * n + 3)) * rpow[n + 3];  // pow<Real>(R[i], -n - 3);
          SVr = a * Vr;
          SVt = a * Vt;
          SVp = a * Vp;

          a = -(2 * n * n + 1) / (Real)((2 * n + 1) * (2 * n - 1)) * rpow[n + 1];  // pow<Real>(R[i], -n - 1);
          b = n * (n + 2) / (Real)(2 * n + 1) * (rpow[n + 1] - rpow[n + 3]);       //(pow<Real>(R[i], -n - 1) - pow<Real>(R[i], -n - 3));
          SWr = a * Wr + b * Vr;
          SWt = a * Wt + b * Vt;
          SWp = a * Wp + b * Vp;

          a = -(n + 2) / (Real)(2 * n + 1) * rpow[n + 2];  // pow<Real>(R[i], -n - 2);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        }

        write_coeff(SVr, n, m, 0, 0);
        write_coeff(SVt, n, m, 0, 1);
        write_coeff(SVp, n, m, 0, 2);

        write_coeff(SWr, n, m, 1, 0);
        write_coeff(SWt, n, m, 1, 1);
        write_coeff(SWp, n, m, 1, 2);

        write_coeff(SXr, n, m, 2, 0);
        write_coeff(SXt, n, m, 2, 1);
        write_coeff(SXp, n, m, 2, 2);
      }
    }
  }

  { // Set X <-- Q * StokesOp * B1
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      StaticArray<Real,9> Q;
      { // Set Q
        Real cos_theta = cos(theta_phi[k0 * 2 + 0]);
        Real sin_theta = sin(theta_phi[k0 * 2 + 0]);
        Real cos_phi = cos(theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(theta_phi[k0 * 2 + 1]);
        Q[0] = sin_theta*cos_phi; Q[1] = sin_theta*sin_phi; Q[2] = cos_theta;
        Q[3] = cos_theta*cos_phi; Q[4] = cos_theta*sin_phi; Q[5] =-sin_theta;
        Q[6] =          -sin_phi; Q[7] =           cos_phi; Q[8] =         0;
      }
      for (Long k1 = 0; k1 < dof; k1++) { // Set X <-- Q * StokesOp * B1
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * StokesOp[k0 * COORD_DIM + j][i];
          }
        }
        X[(k0 * dof + k1) * COORD_DIM + 0] = Q[0] * in[0] + Q[3] * in[1] + Q[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = Q[1] * in[0] + Q[4] * in[1] + Q[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = Q[2] * in[0] + Q[5] * in[1] + Q[8] * in[2];
      }
    }
  }
}





template <class Real> void SphericalHarmonics<Real>::Grid2SHC_(const Vector<Real>& X, Long Nt, Long Np, Long p1, Vector<Real>& B1){
  const auto& Mf = OpFourierInv(Np);
  assert(Mf.Dim(0) == Np);

  const std::vector<Matrix<Real>>& Ml = SphericalHarmonics<Real>::MatLegendreInv(Nt-1,p1);
  assert((Long)Ml.size() == p1+1);

  Long N = X.Dim() / (Np*Nt);
  assert(X.Dim() == N*Np*Nt);

  Vector<Real> B0((2*p1+1) * N*Nt);
  #pragma omp parallel
  { // B0 <-- Transpose(FFT(X))
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();
    Long a=(tid+0)*N*Nt/omp_p;
    Long b=(tid+1)*N*Nt/omp_p;

    Vector<Real> buff(Mf.Dim(1));
    Long fft_coeff_len = std::min(buff.Dim(), 2*p1+2);
    Matrix<Real> B0_(2*p1+1, N*Nt, B0.begin(), false);
    const Matrix<Real> MX(N * Nt, Np, (Iterator<Real>)X.begin(), false);
    for (Long i = a; i < b; i++) {
      { // buff <-- FFT(Xi)
        const Vector<Real> Xi(Np, (Iterator<Real>)X.begin() + Np * i, false);
        Mf.Execute(Xi, buff);
      }
      { // B0 <-- Transpose(buff)
        B0_[0][i] = buff[0]; // skipping buff[1] == 0
        for (Long j = 2; j < fft_coeff_len; j++) B0_[j-1][i] = buff[j];
        for (Long j = fft_coeff_len; j < 2*p1+2; j++) B0_[j-1][i] = 0;
      }
    }
  }

  if (B1.Dim() != N*(p1+1)*(p1+1)) B1.ReInit(N*(p1+1)*(p1+1));
  #pragma omp parallel
  { // Evaluate Legendre polynomial
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long offset0=0;
    Long offset1=0;
    for (Long i = 0; i < p1+1; i++) {
      Long N_ = (i==0 ? N : 2*N);
      Matrix<Real> Min (N_, Nt    , B0.begin()+offset0, false);
      Matrix<Real> Mout(N_, p1+1-i, B1.begin()+offset1, false);
      { // Mout = Min * Ml[i]  // split between threads
        Long a=(tid+0)*N_/omp_p;
        Long b=(tid+1)*N_/omp_p;
        if (a < b) {
          Matrix<Real> Min_ (b-a, Min .Dim(1), Min [a], false);
          Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
          Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
        }
      }
      offset0+=Min .Dim(0)*Min .Dim(1);
      offset1+=Mout.Dim(0)*Mout.Dim(1);
    }
    assert(offset0 == B0.Dim());
    assert(offset1 == B1.Dim());
  }
  B1 *= 1 / sqrt<Real>(4 * const_pi<Real>() * Np); // Scaling to match Zydrunas Fortran code.
}
template <class Real> void SphericalHarmonics<Real>::SHCArrange0(const Vector<Real>& B1, Long p1, Vector<Real>& S, SHCArrange arrange){
  Long M = (p1+1)*(p1+1);
  Long N = B1.Dim() / M;
  assert(B1.Dim() == N*M);
  if (arrange == SHCArrange::ALL) { // S <-- Rearrange(B1)
    Long M = 2*(p1+1)*(p1+1);
    if(S.Dim() != N * M) S.ReInit(N * M);
    #pragma omp parallel
    { // S <-- Rearrange(B1)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p1+1; j++) {
          Long len = p1+1 - j;
          if (1) { // Set Real(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + j*(p1+1)*2 + j*2 + 0;
            for (Long k = 0; k < len; k++) S_[k * (p1+1)*2] = B_[k];
            offset += len;
          }
          if (j) { // Set Imag(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + j*(p1+1)*2 + j*2 + 1;
            for (Long k = 0; k < len; k++) S_[k * (p1+1)*2] = B_[k];
            offset += len;
          } else {
            Iterator<Real>      S_ = S .begin() + i*M   + j*(p1+1)*2 + j*2 + 1;
            for (Long k = 0; k < len; k++) S_[k * (p1+1)*2] = 0;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::ROW_MAJOR) { // S <-- Rearrange(B1)
    Long M = (p1+1)*(p1+2);
    if(S.Dim() != N * M) S.ReInit(N * M);
    #pragma omp parallel
    { // S <-- Rearrange(B1)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p1+1; j++) {
          Long len = p1+1 - j;
          if (1) { // Set Real(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M + 0;
            for (Long k=0;k<len;k++) S_[(j+k)*(j+k+1) + 2*j] = B_[k];
            offset += len;
          }
          if (j) { // Set Imag(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M + 1;
            for (Long k=0;k<len;k++) S_[(j+k)*(j+k+1) + 2*j] = B_[k];
            offset += len;
          } else {
            Iterator<Real> S_ = S .begin() + i*M + 1;
            for (Long k=0;k<len;k++) S_[(j+k)*(j+k+1) + 2*j] = 0;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::COL_MAJOR_NONZERO) { // S <-- Rearrange(B1)
    Long M = (p1+1)*(p1+1);
    if(S.Dim() != N * M) S.ReInit(N * M);
    #pragma omp parallel
    { // S <-- Rearrange(B1)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j <  p1+1; j++) {
          Long len = p1+1 - j;
          if (1) { // Set Real(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) S_[k] = B_[k];
            offset += len;
          }
          if (j) { // Set Imag(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) S_[k] = B_[k];
            offset += len;
          }
        }
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::SHCArrange1(const Vector<Real>& S, SHCArrange arrange, Long p0, Vector<Real>& B0){
  Long M, N;
  { // Set M, N
    M = 0;
    if (arrange == SHCArrange::ALL) M = 2*(p0+1)*(p0+1);
    if (arrange == SHCArrange::ROW_MAJOR) M = (p0+1)*(p0+2);
    if (arrange == SHCArrange::COL_MAJOR_NONZERO) M = (p0+1)*(p0+1);
    if (M == 0) return;
    N = S.Dim() / M;
    assert(S.Dim() == N * M);
  }

  if (B0.Dim() != N*(p0+1)*(p0+1)) B0.ReInit(N*(p0+1)*(p0+1));
  if (arrange == SHCArrange::ALL) { // B0 <-- Rearrange(S)
    #pragma omp parallel
    { // B0 <-- Rearrange(S)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p0+1; j++) {
          Long len = p0+1 - j;
          if (1) { // Get Real(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + j*(p0+1)*2 + j*2 + 0;
            for (Long k = 0; k < len; k++) B_[k] = S_[k * (p0+1)*2];
            offset += len;
          }
          if (j) { // Get Imag(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + j*(p0+1)*2 + j*2 + 1;
            for (Long k = 0; k < len; k++) B_[k] = S_[k * (p0+1)*2];
            offset += len;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::ROW_MAJOR) { // B0 <-- Rearrange(S)
    #pragma omp parallel
    { // B0 <-- Rearrange(S)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p0+1; j++) {
          Long len = p0+1 - j;
          if (1) { // Get Real(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M + 0;
            for (Long k=0;k<len;k++) B_[k] = S_[(j+k)*(j+k+1) + 2*j];
            offset += len;
          }
          if (j) { // Get Imag(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M + 1;
            for (Long k=0;k<len;k++) B_[k] = S_[(j+k)*(j+k+1) + 2*j];
            offset += len;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::COL_MAJOR_NONZERO) { // B0 <-- Rearrange(S)
    #pragma omp parallel
    { // B0 <-- Rearrange(S)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j <  p0+1; j++) {
          Long len = p0+1 - j;
          if (1) { // Get Real(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) B_[k] = S_[k];
            offset += len;
          }
          if (j) { // Get Imag(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) B_[k] = S_[k];
            offset += len;
          }
        }
      }
    }
  }
}
template <class Real> void SphericalHarmonics<Real>::SHC2Grid_(const Vector<Real>& B0, Long p0, Long Nt, Long Np, Vector<Real>* X, Vector<Real>* X_phi, Vector<Real>* X_theta){
  const auto& Mf = OpFourier(Np);
  assert(Mf.Dim(1) == Np);

  const std::vector<Matrix<Real>>& Ml =SphericalHarmonics<Real>::MatLegendre    (p0,Nt-1);
  const std::vector<Matrix<Real>>& Mdl=SphericalHarmonics<Real>::MatLegendreGrad(p0,Nt-1);
  assert((Long)Ml .size() == p0+1);
  assert((Long)Mdl.size() == p0+1);

  Long N = B0.Dim() / ((p0+1)*(p0+1));
  assert(B0.Dim() == N*(p0+1)*(p0+1));

  if(X       && X      ->Dim()!=N*Np*Nt) X      ->ReInit(N*Np*Nt);
  if(X_theta && X_theta->Dim()!=N*Np*Nt) X_theta->ReInit(N*Np*Nt);
  if(X_phi   && X_phi  ->Dim()!=N*Np*Nt) X_phi  ->ReInit(N*Np*Nt);

  Vector<Real> B1(N*(2*p0+1)*Nt);
  if(X || X_phi){
    #pragma omp parallel
    { // Evaluate Legendre polynomial
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long offset0=0;
      Long offset1=0;
      for(Long i=0;i<p0+1;i++){
        Long N_ = (i==0 ? N : 2*N);
        const Matrix<Real> Min (N_, p0+1-i, (Iterator<Real>)B0.begin()+offset0, false);
        Matrix<Real> Mout(N_, Nt    , B1.begin()+offset1, false);
        { // Mout = Min * Ml[i]  // split between threads
          Long a=(tid+0)*N_/omp_p;
          Long b=(tid+1)*N_/omp_p;
          if(a<b){
            const Matrix<Real> Min_ (b-a, Min .Dim(1), (Iterator<Real>)Min [a], false);
            Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
            Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
          }
        }
        offset0+=Min .Dim(0)*Min .Dim(1);
        offset1+=Mout.Dim(0)*Mout.Dim(1);
      }
    }
    B1 *= sqrt<Real>(4 * const_pi<Real>() * Np); // Scaling to match Zydrunas Fortran code.

    #pragma omp parallel
    { // Transpose and evaluate Fourier
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N*Nt/omp_p;
      Long b=(tid+1)*N*Nt/omp_p;

      Vector<Real> buff(Mf.Dim(0)); buff = 0;
      Long fft_coeff_len = std::min(buff.Dim(), 2*p0+2);
      Matrix<Real> B1_(2*p0+1, N*Nt, B1.begin(), false);
      for (Long i = a; i < b; i++) {
        { // buff <-- Transpose(B1)
          buff[0] = B1_[0][i];
          buff[1] = 0;
          for (Long j = 2; j < fft_coeff_len; j++) buff[j] = B1_[j-1][i];
          for (Long j = fft_coeff_len; j < buff.Dim(); j++) buff[j] = 0;
        }
        { // X <-- FFT(buff)
          Vector<Real> Xi(Np, X->begin() + Np * i, false);
          Mf.Execute(buff, Xi);
        }

        if(X_phi){ // Evaluate Fourier gradient
          { // buff <-- Transpose(B1)
            buff[0] = 0;
            buff[1] = 0;
            for (Long j = 2; j < fft_coeff_len; j++) buff[j] = B1_[j-1][i];
            for (Long j = fft_coeff_len; j < buff.Dim(); j++) buff[j] = 0;
            for (Long j = 1; j < buff.Dim()/2; j++) {
              Real x = buff[2*j+0];
              Real y = buff[2*j+1];
              buff[2*j+0] = -j*y;
              buff[2*j+1] =  j*x;
            }
          }
          { // X_phi <-- FFT(buff)
            Vector<Real> Xi(Np, X_phi->begin() + Np * i, false);
            Mf.Execute(buff, Xi);
          }
        }
      }
    }
  }
  if(X_theta){
    #pragma omp parallel
    { // Evaluate Legendre gradient
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long offset0=0;
      Long offset1=0;
      for(Long i=0;i<p0+1;i++){
        Long N_ = (i==0 ? N : 2*N);
        const Matrix<Real> Min (N_, p0+1-i, (Iterator<Real>)B0.begin()+offset0, false);
        Matrix<Real> Mout(N_, Nt    , B1.begin()+offset1, false);
        { // Mout = Min * Mdl[i]  // split between threads
          Long a=(tid+0)*N_/omp_p;
          Long b=(tid+1)*N_/omp_p;
          if(a<b){
            const Matrix<Real> Min_ (b-a, Min .Dim(1), (Iterator<Real>)Min [a], false);
            Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
            Matrix<Real>::GEMM(Mout_,Min_,Mdl[i]);
          }
        }
        offset0+=Min .Dim(0)*Min .Dim(1);
        offset1+=Mout.Dim(0)*Mout.Dim(1);
      }
    }
    B1 *= sqrt<Real>(4 * const_pi<Real>() * Np); // Scaling to match Zydrunas Fortran code.

    #pragma omp parallel
    { // Transpose and evaluate Fourier
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N*Nt/omp_p;
      Long b=(tid+1)*N*Nt/omp_p;

      Vector<Real> buff(Mf.Dim(0)); buff = 0;
      Long fft_coeff_len = std::min(buff.Dim(), 2*p0+2);
      Matrix<Real> B1_(2*p0+1, N*Nt, B1.begin(), false);
      for (Long i = a; i < b; i++) {
        { // buff <-- Transpose(B1)
          buff[0] = B1_[0][i];
          buff[1] = 0;
          for (Long j = 2; j < fft_coeff_len; j++) buff[j] = B1_[j-1][i];
          for (Long j = fft_coeff_len; j < buff.Dim(); j++) buff[j] = 0;
        }
        { // Xi <-- FFT(buff)
          Vector<Real> Xi(Np, X_theta->begin() + Np * i, false);
          Mf.Execute(buff, Xi);
        }
      }
    }
  }
}


template <class Real> void SphericalHarmonics<Real>::LegPoly(Vector<Real>& poly_val, const Vector<Real>& X, Long degree){
  Vector<Real> theta(X.Dim());
  for (Long i = 0; i < X.Dim(); i++) theta[i] = acos(X[i]);
  LegPoly_(poly_val, theta, degree);
}
template <class Real> void SphericalHarmonics<Real>::LegPoly_(Vector<Real>& poly_val, const Vector<Real>& theta, Long degree){
  Long N = theta.Dim();
  Long Npoly = (degree + 1) * (degree + 2) / 2;
  if (poly_val.Dim() != Npoly * N) poly_val.ReInit(Npoly * N);

  Real fact = 1 / sqrt<Real>(4 * const_pi<Real>());
  Vector<Real> cos_theta(N), sin_theta(N);
  for (Long n = 0; n < N; n++) {
    cos_theta[n] = cos(theta[n]);
    sin_theta[n] = sin(theta[n]);
    poly_val[n] = fact;
  }

  Long idx = 0;
  Long idx_nxt = 0;
  for (Long i = 1; i <= degree; i++) {
    idx_nxt += N*(degree-i+2);
    Real c = sqrt<Real>((2*i+1)/(Real)(2*i));
    for (Long n = 0; n < N; n++) {
      poly_val[idx_nxt+n] = -poly_val[idx+n] * sin_theta[n] * c;
    }
    idx = idx_nxt;
  }

  idx = 0;
  for (Long m = 0; m < degree; m++) {
    for (Long n = 0; n < N; n++) {
      Real pmm = 0;
      Real pmmp1 = poly_val[idx+n];
      for (Long ll = m + 1; ll <= degree; ll++) {
        Real a = sqrt<Real>(((2*ll-1)*(2*ll+1)         ) / (Real)((ll-m)*(ll+m)         ));
        Real b = sqrt<Real>(((2*ll+1)*(ll+m-1)*(ll-m-1)) / (Real)((ll-m)*(ll+m)*(2*ll-3)));
        Real pll = cos_theta[n]*a*pmmp1 - b*pmm;
        pmm = pmmp1;
        pmmp1 = pll;
        poly_val[idx + N*(ll-m) + n] = pll;
      }
    }
    idx += N * (degree - m + 1);
  }
}

template <class Real> void SphericalHarmonics<Real>::LegPolyDeriv(Vector<Real>& poly_val, const Vector<Real>& X, Long degree){
  Vector<Real> theta(X.Dim());
  for (Long i = 0; i < X.Dim(); i++) theta[i] = acos(X[i]);
  LegPolyDeriv_(poly_val, theta, degree);
}
template <class Real> void SphericalHarmonics<Real>::LegPolyDeriv_(Vector<Real>& poly_val, const Vector<Real>& theta, Long degree){
  Long N = theta.Dim();
  Long Npoly = (degree + 1) * (degree + 2) / 2;
  if (poly_val.Dim() != N * Npoly) poly_val.ReInit(N * Npoly);

  Vector<Real> cos_theta(N), sin_theta(N);
  for (Long i = 0; i < N; i++) {
    cos_theta[i] = cos(theta[i]);
    sin_theta[i] = sin(theta[i]);
  }

  Vector<Real> leg_poly(Npoly * N);
  LegPoly_(leg_poly, theta, degree);

  for (Long m = 0; m <= degree; m++) {
    for (Long n = m; n <= degree; n++) {
      ConstIterator<Real> Pn  = leg_poly.begin() + N * ((degree * 2 - m + 1) * (m + 0) / 2 + n);
      ConstIterator<Real> Pn_ = leg_poly.begin() + N * ((degree * 2 - m + 0) * (m + 1) / 2 + n) * (m < n);
      Iterator     <Real> Hn  = poly_val.begin() + N * ((degree * 2 - m + 1) * (m + 0) / 2 + n);

      Real c2 = sqrt<Real>(m<n ? (n+m+1)*(n-m) : 0);
      for (Long i = 0; i < N; i++) {
        Real c1 = (sin_theta[i]>0 ? m/sin_theta[i] : 0);
        Hn[i] = c1*cos_theta[i]*Pn[i] + c2*Pn_[i];
      }
    }
  }
}


template <class Real> const Vector<Real>& SphericalHarmonics<Real>::LegendreNodes(Long p){
  assert(p<SCTL_SHMAXDEG);
  Vector<Real>& Qx=MatrixStore().Qx_[p];
  #pragma omp critical (SCTL_LEGNODES)
  if(!Qx.Dim()){
    Vector<double> qx1(p+1);
    Vector<double> qw1(p+1);
    cgqf(p+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);
    assert(typeid(Real) == typeid(double) || typeid(Real) == typeid(float)); // TODO: works only for float and double
    if (Qx.Dim() != p+1) Qx.ReInit(p+1);
    for (Long i = 0; i < p + 1; i++) Qx[i] = -qx1[i];
  }
  return Qx;
}

template <class Real> const Vector<Real>& SphericalHarmonics<Real>::LegendreWeights(Long p){
  assert(p<SCTL_SHMAXDEG);
  Vector<Real>& Qw=MatrixStore().Qw_[p];
  #pragma omp critical (SCTL_LEGWEIGHTS)
  if(!Qw.Dim()){
    Vector<double> qx1(p+1);
    Vector<double> qw1(p+1);
    cgqf(p+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);
    assert(typeid(Real) == typeid(double) || typeid(Real) == typeid(float)); // TODO: works only for float and double
    if (Qw.Dim() != p+1) Qw.ReInit(p+1);
    for (Long i = 0; i < p + 1; i++) Qw[i] = qw1[i];
  }
  return Qw;
}

template <class Real> const Vector<Real>& SphericalHarmonics<Real>::SingularWeights(Long p1){
  assert(p1<SCTL_SHMAXDEG);
  Vector<Real>& Sw=MatrixStore().Sw_[p1];
  #pragma omp critical (SCTL_SINWEIGHTS)
  if(!Sw.Dim()){
    const Vector<Real>& qx1 = LegendreNodes(p1);
    const Vector<Real>& qw1 = LegendreWeights(p1);

    std::vector<Real> Yf(p1+1,0);
    { // Set Yf
      Vector<Real> x0(1); x0=1.0;
      Vector<Real> alp0((p1+1)*(p1+2)/2);
      LegPoly(alp0, x0, p1);

      Vector<Real> alp((p1+1) * (p1+1)*(p1+2)/2);
      LegPoly(alp, qx1, p1);

      for(Long j=0;j<p1+1;j++){
        for(Long i=0;i<p1+1;i++){
          Yf[i]+=4*M_PI/(2*j+1) * alp0[j] * alp[j*(p1+1)+i];
        }
      }
    }

    Sw.ReInit(p1+1);
    for(Long i=0;i<p1+1;i++){
      Sw[i]=(qw1[i]*M_PI/p1)*Yf[i]/cos(acos(qx1[i])/2);
    }
  }
  return Sw;
}


template <class Real> const Matrix<Real>& SphericalHarmonics<Real>::MatFourier(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  Matrix<Real>& Mf =MatrixStore().Mf_ [p0*SCTL_SHMAXDEG+p1];
  #pragma omp critical (SCTL_MATFOURIER)
  if(!Mf.Dim(0)){
    const Real SQRT2PI=sqrt(2*M_PI);
    { // Set Mf
      Matrix<Real> M(2*p0,2*p1);
      for(Long j=0;j<2*p1;j++){
        M[0][j]=SQRT2PI*1.0;
        for(Long k=1;k<p0;k++){
          M[2*k-1][j]=SQRT2PI*cos(j*k*M_PI/p1);
          M[2*k-0][j]=SQRT2PI*sin(j*k*M_PI/p1);
        }
        M[2*p0-1][j]=SQRT2PI*cos(j*p0*M_PI/p1);
      }
      Mf=M;
    }
  }
  return Mf;
}

template <class Real> const Matrix<Real>& SphericalHarmonics<Real>::MatFourierInv(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  Matrix<Real>& Mf =MatrixStore().Mfinv_ [p0*SCTL_SHMAXDEG+p1];
  #pragma omp critical (SCTL_MATFOURIERINV)
  if(!Mf.Dim(0)){
    const Real INVSQRT2PI=1.0/sqrt(2*M_PI)/p0;
    { // Set Mf
      Matrix<Real> M(2*p0,2*p1);
      M.SetZero();
      if(p1>p0) p1=p0;
      for(Long j=0;j<2*p0;j++){
        M[j][0]=INVSQRT2PI*0.5;
        for(Long k=1;k<p1;k++){
          M[j][2*k-1]=INVSQRT2PI*cos(j*k*M_PI/p0);
          M[j][2*k-0]=INVSQRT2PI*sin(j*k*M_PI/p0);
        }
        M[j][2*p1-1]=INVSQRT2PI*cos(j*p1*M_PI/p0);
      }
      if(p1==p0) for(Long j=0;j<2*p0;j++) M[j][2*p1-1]*=0.5;
      Mf=M;
    }
  }
  return Mf;
}

template <class Real> const Matrix<Real>& SphericalHarmonics<Real>::MatFourierGrad(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  Matrix<Real>& Mdf=MatrixStore().Mdf_[p0*SCTL_SHMAXDEG+p1];
  #pragma omp critical (SCTL_MATFOURIERGRAD)
  if(!Mdf.Dim(0)){
    const Real SQRT2PI=sqrt(2*M_PI);
    { // Set Mdf_
      Matrix<Real> M(2*p0,2*p1);
      for(Long j=0;j<2*p1;j++){
        M[0][j]=SQRT2PI*0.0;
        for(Long k=1;k<p0;k++){
          M[2*k-1][j]=-SQRT2PI*k*sin(j*k*M_PI/p1);
          M[2*k-0][j]= SQRT2PI*k*cos(j*k*M_PI/p1);
        }
        M[2*p0-1][j]=-SQRT2PI*p0*sin(j*p0*M_PI/p1);
      }
      Mdf=M;
    }
  }
  return Mdf;
}


template <class Real> const FFT<Real>& SphericalHarmonics<Real>::OpFourier(Long Np){
  assert(Np<SCTL_SHMAXDEG);
  auto& Mf =MatrixStore().Mfftinv_ [Np];
  #pragma omp critical (SCTL_FFT_PLAN0)
  if(!Mf.Dim(0)){
    StaticArray<Long,1> fft_dim{Np};
    Mf.Setup(FFT_Type::C2R, 1, Vector<Long>(1,fft_dim,false));
  }
  return Mf;
}

template <class Real> const FFT<Real>& SphericalHarmonics<Real>::OpFourierInv(Long Np){
  assert(Np<SCTL_SHMAXDEG);
  auto& Mf =MatrixStore().Mfft_ [Np];
  #pragma omp critical (SCTL_FFT_PLAN1)
  if(!Mf.Dim(0)){
    StaticArray<Long,1> fft_dim {Np};
    Mf.Setup(FFT_Type::R2C, 1, Vector<Long>(1,fft_dim,false));
  }
  return Mf;
}


template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendre(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Ml =MatrixStore().Ml_ [p0*SCTL_SHMAXDEG+p1];
  #pragma omp critical (SCTL_MATLEG)
  if(!Ml.size()){
    const Vector<Real>& qx1 = LegendreNodes(p1);
    Vector<Real> alp(qx1.Dim()*(p0+1)*(p0+2)/2);
    LegPoly(alp, qx1, p0);

    Ml.resize(p0+1);
    auto ptr = alp.begin();
    for(Long i=0;i<=p0;i++){
      Ml[i].ReInit(p0+1-i, qx1.Dim(), ptr);
      ptr+=Ml[i].Dim(0)*Ml[i].Dim(1);
    }
  }
  return Ml;
}

template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendreInv(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Ml =MatrixStore().Mlinv_ [p0*SCTL_SHMAXDEG+p1];
  #pragma omp critical (SCTL_MATLEGINV)
  if(!Ml.size()){
    const Vector<Real>& qx1 = LegendreNodes(p0);
    const Vector<Real>& qw1 = LegendreWeights(p0);
    Vector<Real> alp(qx1.Dim()*(p1+1)*(p1+2)/2);
    LegPoly(alp, qx1, p1);

    Ml.resize(p1+1);
    auto ptr = alp.begin();
    for(Long i=0;i<=p1;i++){
      Ml[i].ReInit(qx1.Dim(), p1+1-i);
      Matrix<Real> M(p1+1-i, qx1.Dim(), ptr, false);
      for(Long j=0;j<p1+1-i;j++){ // Transpose and weights
        for(Long k=0;k<qx1.Dim();k++){
          Ml[i][k][j]=M[j][k]*qw1[k]*2*M_PI;
        }
      }
      ptr+=Ml[i].Dim(0)*Ml[i].Dim(1);
    }
  }
  return Ml;
}

template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendreGrad(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Mdl=MatrixStore().Mdl_[p0*SCTL_SHMAXDEG+p1];
  #pragma omp critical (SCTL_MATLEGGRAD)
  if(!Mdl.size()){
    const Vector<Real>& qx1 = LegendreNodes(p1);
    Vector<Real> alp(qx1.Dim()*(p0+1)*(p0+2)/2);
    LegPolyDeriv(alp, qx1, p0);

    Mdl.resize(p0+1);
    auto ptr = alp.begin();
    for(Long i=0;i<=p0;i++){
      Mdl[i].ReInit(p0+1-i, qx1.Dim(), ptr);
      ptr+=Mdl[i].Dim(0)*Mdl[i].Dim(1);
    }
  }
  return Mdl;
}


template <class Real> void SphericalHarmonics<Real>::SHBasisEval(Long p0, const Vector<Real>& theta_phi, Matrix<Real>& SHBasis) {
  Long M = (p0+1) * (p0+1);
  Long N = theta_phi.Dim() / 2;
  assert(theta_phi.Dim() == N * 2);

  Vector<Complex<Real>> exp_phi(N);
  Matrix<Real> LegP((p0+1)*(p0+2)/2, N);
  { // Set exp_phi, LegP
    Vector<Real> theta(N);
    for (Long i = 0; i < N; i++) { // Set theta, exp_phi
      theta[i] = theta_phi[i*2+0];
      exp_phi[i].real = cos<Real>(theta_phi[i*2+1]);
      exp_phi[i].imag = sin<Real>(theta_phi[i*2+1]);
    }

    Vector<Real> alp(LegP.Dim(0) * LegP.Dim(1), LegP.begin(), false);
    LegPoly_(alp, theta, p0);
  }

  { // Set SHBasis
    SHBasis.ReInit(N, M);
    Real s = 4 * sqrt<Real>(const_pi<Real>());
    for (Long k0 = 0; k0 < N; k0++) {
      Complex<Real> exp_phi_ = 1;
      Complex<Real> exp_phi1 = exp_phi[k0];
      for (Long m = 0; m <= p0; m++) {
        for (Long n = m; n <= p0; n++) {
          Long poly_idx = (2 * p0 - m + 1) * m / 2 + n;
          Long basis_idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
          SHBasis[k0][basis_idx] = LegP[poly_idx][k0] * exp_phi_.real * s;
          if (m) { // imaginary part
            basis_idx += (p0+1-m);
            SHBasis[k0][basis_idx] = -LegP[poly_idx][k0] * exp_phi_.imag * s;
          } else {
            SHBasis[k0][basis_idx] = SHBasis[k0][basis_idx] * 0.5;
          }
        }
        exp_phi_ = exp_phi_ * exp_phi1;
      }
    }
  }
  assert(SHBasis.Dim(0) == N);
  assert(SHBasis.Dim(1) == M);
}

template <class Real> void SphericalHarmonics<Real>::VecSHBasisEval(Long p0, const Vector<Real>& theta_phi, Matrix<Real>& SHBasis) {
  Long M = (p0+1) * (p0+1);
  Long N = theta_phi.Dim() / 2;
  assert(theta_phi.Dim() == N * 2);

  Long p_ = p0 + 1;
  Long M_ = (p_+1) * (p_+1);
  Matrix<Real> Ynm(N, M_);
  SHBasisEval(p_, theta_phi, Ynm);

  Vector<Real> cos_theta(N), csc_theta(N);
  for (Long i = 0; i < N; i++) { // Set theta
    cos_theta[i] = cos(theta_phi[i*2+0]);
    csc_theta[i] = 1.0 / sin(theta_phi[i*2+0]);
  }

  { // Set SHBasis
    SHBasis.ReInit(N * COORD_DIM, COORD_DIM * M);
    SHBasis = 0;
    const Complex<Real> imag(0,1);
    for (Long i = 0; i < N; i++) {
      auto Y = [p_, &Ynm, i](Long n, Long m) {
        Complex<Real> c;
        if (0 <= m && m <= n && n <= p_) {
          Long idx = (2 * p_ - m + 2) * m - (m ? p_+1 : 0) + n;
          c.real = Ynm[i][idx];
          if (m) {
            idx += (p_+1-m);
            c.imag = Ynm[i][idx];
          }
        }
        return c;
      };
      auto write_coeff = [p0, &SHBasis, i, M](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
        if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
          Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
          SHBasis[i * COORD_DIM + k1][k0 * M + idx] = c.real;
          if (m) {
            idx += (p0+1-m);
            SHBasis[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
          }
        }
      };
      auto A = [p_](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>(n*n * ((n+1)*(n+1) - m*m) / (Real)((2*n+1)*(2*n+3))) : 0); };
      auto B = [p_](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>((n+1)*(n+1) * (n*n - m*m) / (Real)((2*n+1)*(2*n-1))) : 0); };
      if (fabs(csc_theta[i]) > 0) {
        for (Long m = 0; m <= p0; m++) {
          for (Long n = m; n <= p0; n++) {
            Complex<Real> AYBY = A(n,m) * Y(n+1,m) - B(n,m) * Y(n-1,m);

            Complex<Real> Fv2r = Y(n,m) * (-n-1);
            Complex<Real> Fw2r = Y(n,m) * n;
            Complex<Real> Fx2r = 0;

            Complex<Real> Fv2t = AYBY * csc_theta[i];
            Complex<Real> Fw2t = AYBY * csc_theta[i];
            Complex<Real> Fx2t = imag * m * Y(n,m) * csc_theta[i];

            Complex<Real> Fv2p = -imag * m * Y(n,m) * csc_theta[i];
            Complex<Real> Fw2p = -imag * m * Y(n,m) * csc_theta[i];
            Complex<Real> Fx2p = AYBY * csc_theta[i];

            write_coeff(Fv2r, n, m, 0, 0);
            write_coeff(Fw2r, n, m, 1, 0);
            write_coeff(Fx2r, n, m, 2, 0);

            write_coeff(Fv2t, n, m, 0, 1);
            write_coeff(Fw2t, n, m, 1, 1);
            write_coeff(Fx2t, n, m, 2, 1);

            write_coeff(Fv2p, n, m, 0, 2);
            write_coeff(Fw2p, n, m, 1, 2);
            write_coeff(Fx2p, n, m, 2, 2);
          }
        }
      } else {
        Complex<Real> exp_phi;
        exp_phi.real = cos<Real>(theta_phi[i*2+1]);
        exp_phi.imag = -sin<Real>(theta_phi[i*2+1]);
        for (Long m = 0; m <= p0; m++) {
          for (Long n = m; n <= p0; n++) {

            Complex<Real> Fv2r = 0;
            Complex<Real> Fw2r = 0;
            Complex<Real> Fx2r = 0;
            Complex<Real> Fv2t = 0;
            Complex<Real> Fw2t = 0;
            Complex<Real> Fx2t = 0;
            Complex<Real> Fv2p = 0;
            Complex<Real> Fw2p = 0;
            Complex<Real> Fx2p = 0;

            if (m == 0) {
              Fv2r = Y(n,m) * (-n-1);
              Fw2r = Y(n,m) * n;
              Fx2r = 0;
            }
            if (m == 1) {
              auto Ycsc = [&cos_theta, &exp_phi, i](Long n) { return -sqrt<Real>((2*n+1)*n*(n+1)) * ((n%2==0) && (cos_theta[i]<0) ? -1 : 1) * exp_phi; };
              Complex<Real> AYBY = A(n,m) * Ycsc(n+1) - B(n,m) * Ycsc(n-1);

              Fv2t = AYBY;
              Fw2t = AYBY;
              Fx2t = imag * m * Ycsc(n);

              Fv2p =-imag * m * Ycsc(n);
              Fw2p =-imag * m * Ycsc(n);
              Fx2p = AYBY;
            }

            write_coeff(Fv2r, n, m, 0, 0);
            write_coeff(Fw2r, n, m, 1, 0);
            write_coeff(Fx2r, n, m, 2, 0);

            write_coeff(Fv2t, n, m, 0, 1);
            write_coeff(Fw2t, n, m, 1, 1);
            write_coeff(Fx2t, n, m, 2, 1);

            write_coeff(Fv2p, n, m, 0, 2);
            write_coeff(Fw2p, n, m, 1, 2);
            write_coeff(Fx2p, n, m, 2, 2);
          }
        }
      }
    }
  }
  assert(SHBasis.Dim(0) == N * COORD_DIM);
  assert(SHBasis.Dim(1) == COORD_DIM * M);
}


template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatRotate(Long p0){
  std::vector<std::vector<Long>> coeff_perm(p0+1);
  { // Set coeff_perm
    for(Long n=0;n<=p0;n++) coeff_perm[n].resize(std::min(2*n+1,2*p0));
    Long itr=0;
    for(Long i=0;i<2*p0;i++){
      Long m=(i+1)/2;
      for(Long n=m;n<=p0;n++){
        coeff_perm[n][i]=itr;
        itr++;
      }
    }
  }

  assert(p0<SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Mr=MatrixStore().Mr_[p0];
  #pragma omp critical (SCTL_MATROTATE)
  if(!Mr.size()){
    const Real SQRT2PI=sqrt(2*M_PI);
    Long Ncoef=p0*(p0+2);
    Long Ngrid=2*p0*(p0+1);
    Long Naleg=(p0+1)*(p0+2)/2;

    Matrix<Real> Mcoord0(3,Ngrid);
    const Vector<Real>& x=LegendreNodes(p0);
    for(Long i=0;i<p0+1;i++){ // Set Mcoord0
      for(Long j=0;j<2*p0;j++){
        Mcoord0[0][i*2*p0+j]=x[i];
        Mcoord0[1][i*2*p0+j]=sqrt(1-x[i]*x[i])*sin(M_PI*j/p0);
        Mcoord0[2][i*2*p0+j]=sqrt(1-x[i]*x[i])*cos(M_PI*j/p0);
      }
    }

    for(Long l=0;l<p0+1;l++){ // For each rotation angle
      Matrix<Real> Mcoord1;
      { // Rotate coordinates
        Matrix<Real> M(COORD_DIM, COORD_DIM);
        Real cos_=-x[l];
        Real sin_=-sqrt(1.0-x[l]*x[l]);
        M[0][0]= cos_; M[0][1]=0; M[0][2]=-sin_;
        M[1][0]=    0; M[1][1]=1; M[1][2]=    0;
        M[2][0]= sin_; M[2][1]=0; M[2][2]= cos_;
        Mcoord1=M*Mcoord0;
      }

      Matrix<Real> Mleg(Naleg, Ngrid);
      { // Set Mleg
        const Vector<Real> Vcoord1(Mcoord1.Dim(0)*Mcoord1.Dim(1), Mcoord1.begin(), false);
        Vector<Real> Vleg(Mleg.Dim(0)*Mleg.Dim(1), Mleg.begin(), false);
        LegPoly(Vleg, Vcoord1, p0);
      }

      Vector<Real> theta(Ngrid);
      for(Long i=0;i<theta.Dim();i++){ // Set theta
        theta[i]=atan2(Mcoord1[1][i],Mcoord1[2][i]); // TODO: works only for float and double
      }

      Matrix<Real> Mcoef2grid(Ncoef, Ngrid);
      { // Build Mcoef2grid
        Long offset0=0;
        Long offset1=0;
        for(Long i=0;i<p0+1;i++){
          Long len=p0+1-i;
          { // P * cos
            for(Long j=0;j<len;j++){
              for(Long k=0;k<Ngrid;k++){
                Mcoef2grid[offset1+j][k]=SQRT2PI*Mleg[offset0+j][k]*cos(i*theta[k]);
              }
            }
            offset1+=len;
          }
          if(i!=0 && i!=p0){ // P * sin
            for(Long j=0;j<len;j++){
              for(Long k=0;k<Ngrid;k++){
                Mcoef2grid[offset1+j][k]=SQRT2PI*Mleg[offset0+j][k]*sin(i*theta[k]);
              }
            }
            offset1+=len;
          }
          offset0+=len;
        }
        assert(offset0==Naleg);
        assert(offset1==Ncoef);
      }

      Vector<Real> Vcoef2coef(Ncoef*Ncoef);
      Vector<Real> Vcoef2grid(Ncoef*Ngrid, Mcoef2grid[0], false);
      Grid2SHC(Vcoef2grid, p0+1, 2*p0, p0, Vcoef2coef, SHCArrange::COL_MAJOR_NONZERO);

      Matrix<Real> Mcoef2coef(Ncoef, Ncoef, Vcoef2coef.begin(), false);
      for(Long n=0;n<=p0;n++){ // Create matrices for fast rotation
        Matrix<Real> M(coeff_perm[n].size(),coeff_perm[n].size());
        for(Long i=0;i<(Long)coeff_perm[n].size();i++){
          for(Long j=0;j<(Long)coeff_perm[n].size();j++){
            M[i][j]=Mcoef2coef[coeff_perm[n][i]][coeff_perm[n][j]];
          }
        }
        Mr.push_back(M);
      }
    }
  }
  return Mr;
}



template <class Real> void SphericalHarmonics<Real>::SHC2GridTranspose(const Vector<Real>& X, Long p0, Long p1, Vector<Real>& S){
  Matrix<Real> Mf =SphericalHarmonics<Real>::MatFourier(p1,p0).Transpose();
  std::vector<Matrix<Real>> Ml =SphericalHarmonics<Real>::MatLegendre(p1,p0);
  for(Long i=0;i<(Long)Ml.size();i++) Ml[i]=Ml[i].Transpose();
  assert(p1==(Long)Ml.size()-1);
  assert(p0==Mf.Dim(0)/2);
  assert(p1==Mf.Dim(1)/2);

  Long N=X.Dim()/(2*p0*(p0+1));
  assert(N*2*p0*(p0+1)==X.Dim());
  if(S.Dim()!=N*(p1*(p1+2))) S.ReInit(N*(p1*(p1+2)));

  Vector<Real> B0, B1;
  B0.ReInit(N*  p1*(p1+2));
  B1.ReInit(N*2*p1*(p0+1));

  #pragma omp parallel
  { // Evaluate Fourier and transpose
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N*(p0+1)/omp_p;
    Long b=(tid+1)*N*(p0+1)/omp_p;

    const Long block_size=16;
    Matrix<Real> B2(block_size,2*p1);
    for(Long i0=a;i0<b;i0+=block_size){
      Long i1=std::min(b,i0+block_size);
      const Matrix<Real> Min (i1-i0,2*p0, (Iterator<Real>)X.begin()+i0*2*p0, false);
      Matrix<Real> Mout(i1-i0,2*p1, B2.begin(), false);
      Matrix<Real>::GEMM(Mout, Min, Mf);

      for(Long i=i0;i<i1;i++){
        for(Long j=0;j<2*p1;j++){
          B1[j*N*(p0+1)+i]=B2[i-i0][j];
        }
      }
    }
  }

  #pragma omp parallel
  { // Evaluate Legendre polynomial
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long offset0=0;
    Long offset1=0;
    for(Long i=0;i<p1+1;i++){
      Long N0=2*N;
      if(i==0 || i==p1) N0=N;
      Matrix<Real> Min (N0, p0+1  , B1.begin()+offset0, false);
      Matrix<Real> Mout(N0, p1+1-i, B0.begin()+offset1, false);
      { // Mout = Min * Ml[i]  // split between threads
        Long a=(tid+0)*N0/omp_p;
        Long b=(tid+1)*N0/omp_p;
        if(a<b){
          Matrix<Real> Min_ (b-a, Min .Dim(1), Min [a], false);
          Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
          Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
        }
      }
      offset0+=Min .Dim(0)*Min .Dim(1);
      offset1+=Mout.Dim(0)*Mout.Dim(1);
    }
  }

  #pragma omp parallel
  { // S <-- Rearrange(B0)
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      Long offset=0;
      for(Long j=0;j<2*p1;j++){
        Long len=p1+1-(j+1)/2;
        Real* B_=&B0[i*len+N*offset];
        Real* S_=&S[i*p1*(p1+2)+offset];
        for(Long k=0;k<len;k++) S_[k]=B_[k];
        offset+=len;
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::RotateAll(const Vector<Real>& S, Long p0, Long dof, Vector<Real>& S_){
  const std::vector<Matrix<Real>>& Mr=MatRotate(p0);
  std::vector<std::vector<Long>> coeff_perm(p0+1);
  { // Set coeff_perm
    for(Long n=0;n<=p0;n++) coeff_perm[n].resize(std::min(2*n+1,2*p0));
    Long itr=0;
    for(Long i=0;i<2*p0;i++){
      Long m=(i+1)/2;
      for(Long n=m;n<=p0;n++){
        coeff_perm[n][i]=itr;
        itr++;
      }
    }
  }
  Long Ncoef=p0*(p0+2);

  Long N=S.Dim()/Ncoef/dof;
  assert(N*Ncoef*dof==S.Dim());
  if(S_.Dim()!=N*dof*Ncoef*p0*(p0+1)) S_.ReInit(N*dof*Ncoef*p0*(p0+1));
  const Matrix<Real> S0(N*dof, Ncoef, (Iterator<Real>)S.begin(), false);
  Matrix<Real> S1(N*dof*p0*(p0+1), Ncoef, S_.begin(), false);

  #pragma omp parallel
  { // Construct all p0*(p0+1) rotations
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();
    Matrix<Real> B0(dof*p0,Ncoef); // memory buffer

    std::vector<Matrix<Real>> Bi(p0+1), Bo(p0+1); // memory buffers
    for(Long i=0;i<=p0;i++){ // initialize Bi, Bo
      Bi[i].ReInit(dof*p0,coeff_perm[i].size());
      Bo[i].ReInit(dof*p0,coeff_perm[i].size());
    }

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      for(Long d=0;d<dof;d++){
        for(Long j=0;j<p0;j++){
          Long offset=0;
          for(Long k=0;k<p0+1;k++){
            Real r[2]={cos(k*j*M_PI/p0),-sin(k*j*M_PI/p0)}; // exp(i*k*theta)
            Long len=p0+1-k;
            if(k!=0 && k!=p0){
              for(Long l=0;l<len;l++){
                Real x[2];
                x[0]=S0[i*dof+d][offset+len*0+l];
                x[1]=S0[i*dof+d][offset+len*1+l];
                B0[j*dof+d][offset+len*0+l]=x[0]*r[0]-x[1]*r[1];
                B0[j*dof+d][offset+len*1+l]=x[0]*r[1]+x[1]*r[0];
              }
              offset+=2*len;
            }else{
              for(Long l=0;l<len;l++){
                B0[j*dof+d][offset+l]=S0[i*dof+d][offset+l];
              }
              offset+=len;
            }
          }
          assert(offset==Ncoef);
        }
      }
      { // Fast rotation
        for(Long k=0;k<dof*p0;k++){ // forward permutation
          for(Long l=0;l<=p0;l++){
            for(Long j=0;j<(Long)coeff_perm[l].size();j++){
              Bi[l][k][j]=B0[k][coeff_perm[l][j]];
            }
          }
        }
        for(Long t=0;t<=p0;t++){
          for(Long l=0;l<=p0;l++){ // mat-vec
            Matrix<Real>::GEMM(Bo[l],Bi[l],Mr[t*(p0+1)+l]);
          }
          Matrix<Real> Mout(dof*p0,Ncoef, S1[(i*(p0+1)+t)*dof*p0], false);
          for(Long k=0;k<dof*p0;k++){ // reverse permutation
            for(Long l=0;l<=p0;l++){
              for(Long j=0;j<(Long)coeff_perm[l].size();j++){
                Mout[k][coeff_perm[l][j]]=Bo[l][k][j];
              }
            }
          }
        }
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::RotateTranspose(const Vector<Real>& S_, Long p0, Long dof, Vector<Real>& S){
  std::vector<Matrix<Real>> Mr=MatRotate(p0);
  for(Long i=0;i<(Long)Mr.size();i++) Mr[i]=Mr[i].Transpose();
  std::vector<std::vector<Long>> coeff_perm(p0+1);
  { // Set coeff_perm
    for(Long n=0;n<=p0;n++) coeff_perm[n].resize(std::min(2*n+1,2*p0));
    Long itr=0;
    for(Long i=0;i<2*p0;i++){
      Long m=(i+1)/2;
      for(Long n=m;n<=p0;n++){
        coeff_perm[n][i]=itr;
        itr++;
      }
    }
  }
  Long Ncoef=p0*(p0+2);

  Long N=S_.Dim()/Ncoef/dof/(p0*(p0+1));
  assert(N*Ncoef*dof*(p0*(p0+1))==S_.Dim());
  if(S.Dim()!=N*dof*Ncoef*p0*(p0+1)) S.ReInit(N*dof*Ncoef*p0*(p0+1));
  Matrix<Real> S0(N*dof*p0*(p0+1), Ncoef, S.begin(), false);
  const Matrix<Real> S1(N*dof*p0*(p0+1), Ncoef, (Iterator<Real>)S_.begin(), false);

  #pragma omp parallel
  { // Transpose all p0*(p0+1) rotations
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();
    Matrix<Real> B0(dof*p0,Ncoef); // memory buffer

    std::vector<Matrix<Real>> Bi(p0+1), Bo(p0+1); // memory buffers
    for(Long i=0;i<=p0;i++){ // initialize Bi, Bo
      Bi[i].ReInit(dof*p0,coeff_perm[i].size());
      Bo[i].ReInit(dof*p0,coeff_perm[i].size());
    }

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      for(Long t=0;t<p0+1;t++){
        Long idx0=(i*(p0+1)+t)*p0*dof;
        { // Fast rotation
          const Matrix<Real> Min(p0*dof, Ncoef, (Iterator<Real>)S1[idx0], false);
          for(Long k=0;k<dof*p0;k++){ // forward permutation
            for(Long l=0;l<=p0;l++){
              for(Long j=0;j<(Long)coeff_perm[l].size();j++){
                Bi[l][k][j]=Min[k][coeff_perm[l][j]];
              }
            }
          }
          for(Long l=0;l<=p0;l++){ // mat-vec
            Matrix<Real>::GEMM(Bo[l],Bi[l],Mr[t*(p0+1)+l]);
          }
          for(Long k=0;k<dof*p0;k++){ // reverse permutation
            for(Long l=0;l<=p0;l++){
              for(Long j=0;j<(Long)coeff_perm[l].size();j++){
                B0[k][coeff_perm[l][j]]=Bo[l][k][j];
              }
            }
          }
        }
        for(Long j=0;j<p0;j++){
          for(Long d=0;d<dof;d++){
            Long idx1=idx0+j*dof+d;
            Long offset=0;
            for(Long k=0;k<p0+1;k++){
              Real r[2]={cos(k*j*M_PI/p0),sin(k*j*M_PI/p0)}; // exp(i*k*theta)
              Long len=p0+1-k;
              if(k!=0 && k!=p0){
                for(Long l=0;l<len;l++){
                  Real x[2];
                  x[0]=B0[j*dof+d][offset+len*0+l];
                  x[1]=B0[j*dof+d][offset+len*1+l];
                  S0[idx1][offset+len*0+l]=x[0]*r[0]-x[1]*r[1];
                  S0[idx1][offset+len*1+l]=x[0]*r[1]+x[1]*r[0];
                }
                offset+=2*len;
              }else{
                for(Long l=0;l<len;l++){
                  S0[idx1][offset+l]=B0[j*dof+d][offset+l];
                }
                offset+=len;
              }
            }
            assert(offset==Ncoef);
          }
        }
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::StokesSingularInteg(const Vector<Real>& S, Long p0, Long p1, Vector<Real>* SLMatrix, Vector<Real>* DLMatrix){
  Long Ngrid=2*p0*(p0+1);
  Long Ncoef=  p0*(p0+2);
  Long Nves=S.Dim()/(Ngrid*COORD_DIM);
  if(SLMatrix) SLMatrix->ReInit(Nves*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM));
  if(DLMatrix) DLMatrix->ReInit(Nves*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM));

  Long BLOCK_SIZE=(Long)6e9/((3*2*p1*(p1+1))*(3*2*p0*(p0+1))*2*8); // Limit memory usage to 6GB
  BLOCK_SIZE=std::min<Long>(BLOCK_SIZE,omp_get_max_threads());
  BLOCK_SIZE=std::max<Long>(BLOCK_SIZE,1);

  for(Long a=0;a<Nves;a+=BLOCK_SIZE){
    Long b=std::min(a+BLOCK_SIZE, Nves);

    Vector<Real> _SLMatrix, _DLMatrix;
    if(SLMatrix) _SLMatrix.ReInit((b-a)*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), SLMatrix->begin()+a*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), false);
    if(DLMatrix) _DLMatrix.ReInit((b-a)*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), DLMatrix->begin()+a*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), false);
    const Vector<Real> _S        ((b-a)*(Ngrid*COORD_DIM)                  , (Iterator<Real>)S.begin()+a*(Ngrid*COORD_DIM), false);

    if(SLMatrix && DLMatrix) StokesSingularInteg_< true,  true>(_S, p0, p1, _SLMatrix, _DLMatrix);
    else        if(SLMatrix) StokesSingularInteg_< true, false>(_S, p0, p1, _SLMatrix, _DLMatrix);
    else        if(DLMatrix) StokesSingularInteg_<false,  true>(_S, p0, p1, _SLMatrix, _DLMatrix);
  }
}

template <class Real> template <bool SLayer, bool DLayer> void SphericalHarmonics<Real>::StokesSingularInteg_(const Vector<Real>& X0, Long p0, Long p1, Vector<Real>& SL, Vector<Real>& DL){

  Profile::Tic("Rotate");
  Vector<Real> S0, S;
  SphericalHarmonics<Real>::Grid2SHC(X0, p0+1, 2*p0, p0, S0, SHCArrange::COL_MAJOR_NONZERO);
  SphericalHarmonics<Real>::RotateAll(S0, p0, COORD_DIM, S);
  Profile::Toc();


  Profile::Tic("Upsample");
  Vector<Real> X, X_theta, X_phi, trg;
  SphericalHarmonics<Real>::SHC2Grid(S, SHCArrange::COL_MAJOR_NONZERO, p0, p1+1, 2*p1, &X, &X_theta, &X_phi);
  SphericalHarmonics<Real>::SHC2Pole(S, SHCArrange::COL_MAJOR_NONZERO, p0, trg);
  Profile::Toc();


  Profile::Tic("Stokes");
  Vector<Real> SL0, DL0;
  { // Stokes kernel
    //Long M0=2*p0*(p0+1);
    Long M1=2*p1*(p1+1);
    Long N=trg.Dim()/(2*COORD_DIM);
    assert(X.Dim()==M1*COORD_DIM*N);
    if(SLayer && SL0.Dim()!=N*2*6*M1) SL0.ReInit(2*N*6*M1);
    if(DLayer && DL0.Dim()!=N*2*6*M1) DL0.ReInit(2*N*6*M1);
    const Vector<Real>& qw=SphericalHarmonics<Real>::SingularWeights(p1);

    const Real scal_const_dl = 3.0/(4.0*M_PI);
    const Real scal_const_sl = 1.0/(8.0*M_PI);
    static Real eps=-1;
    if(eps<0){
      eps=1;
      while(eps*(Real)0.5+(Real)1.0>1.0) eps*=0.5;
    }

    #pragma omp parallel
    {
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for(Long i=a;i<b;i++){
        for(Long t=0;t<2;t++){
          Real tx, ty, tz;
          { // Read target coordinates
            tx=trg[i*2*COORD_DIM+0*2+t];
            ty=trg[i*2*COORD_DIM+1*2+t];
            tz=trg[i*2*COORD_DIM+2*2+t];
          }

          for(Long j0=0;j0<p1+1;j0++){
            for(Long j1=0;j1<2*p1;j1++){
              Long s=2*p1*j0+j1;

              Real dx, dy, dz;
              { // Compute dx, dy, dz
                dx=tx-X[(i*COORD_DIM+0)*M1+s];
                dy=ty-X[(i*COORD_DIM+1)*M1+s];
                dz=tz-X[(i*COORD_DIM+2)*M1+s];
              }

              Real nx, ny, nz;
              { // Compute source normal
                Real x_theta=X_phi[(i*COORD_DIM+0)*M1+s];
                Real y_theta=X_phi[(i*COORD_DIM+1)*M1+s];
                Real z_theta=X_phi[(i*COORD_DIM+2)*M1+s];

                Real x_phi=X_theta[(i*COORD_DIM+0)*M1+s];
                Real y_phi=X_theta[(i*COORD_DIM+1)*M1+s];
                Real z_phi=X_theta[(i*COORD_DIM+2)*M1+s];

                nx=(y_theta*z_phi-z_theta*y_phi);
                ny=(z_theta*x_phi-x_theta*z_phi);
                nz=(x_theta*y_phi-y_theta*x_phi);
              }

              Real area_elem=1.0;
              if(SLayer){ // Compute area_elem
                area_elem=sqrt(nx*nx+ny*ny+nz*nz);
              }

              Real rinv, rinv2;
              { // Compute rinv, rinv2
                Real r2=dx*dx+dy*dy+dz*dz;
                rinv=1.0/sqrt(r2);
                if(r2<=eps) rinv=0;
                rinv2=rinv*rinv;
              }

              if(DLayer){
                Real rinv5=rinv2*rinv2*rinv;
                Real r_dot_n_rinv5=scal_const_dl*qw[j0*t+(p1-j0)*(1-t)] * (nx*dx+ny*dy+nz*dz)*rinv5;
                DL0[((i*2+t)*6+0)*M1+s]=dx*dx*r_dot_n_rinv5;
                DL0[((i*2+t)*6+1)*M1+s]=dx*dy*r_dot_n_rinv5;
                DL0[((i*2+t)*6+2)*M1+s]=dx*dz*r_dot_n_rinv5;
                DL0[((i*2+t)*6+3)*M1+s]=dy*dy*r_dot_n_rinv5;
                DL0[((i*2+t)*6+4)*M1+s]=dy*dz*r_dot_n_rinv5;
                DL0[((i*2+t)*6+5)*M1+s]=dz*dz*r_dot_n_rinv5;
              }
              if(SLayer){
                Real area_rinv =scal_const_sl*qw[j0*t+(p1-j0)*(1-t)] * area_elem*rinv;
                Real area_rinv2=area_rinv*rinv2;
                SL0[((i*2+t)*6+0)*M1+s]=area_rinv+dx*dx*area_rinv2;
                SL0[((i*2+t)*6+1)*M1+s]=          dx*dy*area_rinv2;
                SL0[((i*2+t)*6+2)*M1+s]=          dx*dz*area_rinv2;
                SL0[((i*2+t)*6+3)*M1+s]=area_rinv+dy*dy*area_rinv2;
                SL0[((i*2+t)*6+4)*M1+s]=          dy*dz*area_rinv2;
                SL0[((i*2+t)*6+5)*M1+s]=area_rinv+dz*dz*area_rinv2;
              }
            }
          }
        }
      }
    }
    Profile::Add_FLOP(20*(2*p1)*(p1+1)*2*N);
    if(SLayer) Profile::Add_FLOP((19+6)*(2*p1)*(p1+1)*2*N);
    if(DLayer) Profile::Add_FLOP( 22   *(2*p1)*(p1+1)*2*N);
  }
  Profile::Toc();


  Profile::Tic("UpsampleTranspose");
  Vector<Real> SL1, DL1;
  SphericalHarmonics<Real>::SHC2GridTranspose(SL0, p1, p0, SL1);
  SphericalHarmonics<Real>::SHC2GridTranspose(DL0, p1, p0, DL1);
  Profile::Toc();


  Profile::Tic("RotateTranspose");
  Vector<Real> SL2, DL2;
  SphericalHarmonics<Real>::RotateTranspose(SL1, p0, 2*6, SL2);
  SphericalHarmonics<Real>::RotateTranspose(DL1, p0, 2*6, DL2);
  Profile::Toc();


  Profile::Tic("Rearrange");
  Vector<Real> SL3, DL3;
  { // Transpose
    Long Ncoef=p0*(p0+2);
    Long Ngrid=2*p0*(p0+1);
    { // Transpose SL2
      Long N=SL2.Dim()/(6*Ncoef*Ngrid);
      SL3.ReInit(N*COORD_DIM*Ncoef*COORD_DIM*Ngrid);
      #pragma omp parallel
      {
        Integer tid=omp_get_thread_num();
        Integer omp_p=omp_get_num_threads();
        Matrix<Real> B(COORD_DIM*Ncoef,Ngrid*COORD_DIM);

        Long a=(tid+0)*N/omp_p;
        Long b=(tid+1)*N/omp_p;
        for(Long i=a;i<b;i++){
          Matrix<Real> M0(Ngrid*6, Ncoef, SL2.begin()+i*Ngrid*6*Ncoef, false);
          for(Long k=0;k<Ncoef;k++){ // Transpose
            for(Long j=0;j<Ngrid;j++){ // TODO: needs blocking
              B[k+Ncoef*0][j*COORD_DIM+0]=M0[j*6+0][k];
              B[k+Ncoef*1][j*COORD_DIM+0]=M0[j*6+1][k];
              B[k+Ncoef*2][j*COORD_DIM+0]=M0[j*6+2][k];
              B[k+Ncoef*0][j*COORD_DIM+1]=M0[j*6+1][k];
              B[k+Ncoef*1][j*COORD_DIM+1]=M0[j*6+3][k];
              B[k+Ncoef*2][j*COORD_DIM+1]=M0[j*6+4][k];
              B[k+Ncoef*0][j*COORD_DIM+2]=M0[j*6+2][k];
              B[k+Ncoef*1][j*COORD_DIM+2]=M0[j*6+4][k];
              B[k+Ncoef*2][j*COORD_DIM+2]=M0[j*6+5][k];
            }
          }
          Matrix<Real> M1(Ncoef*COORD_DIM, COORD_DIM*Ngrid, SL3.begin()+i*COORD_DIM*Ncoef*COORD_DIM*Ngrid, false);
          for(Long k=0;k<B.Dim(0);k++){ // Rearrange
            for(Long j0=0;j0<COORD_DIM;j0++){
              for(Long j1=0;j1<p0+1;j1++){
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+   j1)*2+0)*p0+j2]=B[k][((j1*p0+j2)*2+0)*COORD_DIM+j0];
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+p0-j1)*2+1)*p0+j2]=B[k][((j1*p0+j2)*2+1)*COORD_DIM+j0];
              }
            }
          }
        }
      }
    }
    { // Transpose DL2
      Long N=DL2.Dim()/(6*Ncoef*Ngrid);
      DL3.ReInit(N*COORD_DIM*Ncoef*COORD_DIM*Ngrid);
      #pragma omp parallel
      {
        Integer tid=omp_get_thread_num();
        Integer omp_p=omp_get_num_threads();
        Matrix<Real> B(COORD_DIM*Ncoef,Ngrid*COORD_DIM);

        Long a=(tid+0)*N/omp_p;
        Long b=(tid+1)*N/omp_p;
        for(Long i=a;i<b;i++){
          Matrix<Real> M0(Ngrid*6, Ncoef, DL2.begin()+i*Ngrid*6*Ncoef, false);
          for(Long k=0;k<Ncoef;k++){ // Transpose
            for(Long j=0;j<Ngrid;j++){ // TODO: needs blocking
              B[k+Ncoef*0][j*COORD_DIM+0]=M0[j*6+0][k];
              B[k+Ncoef*1][j*COORD_DIM+0]=M0[j*6+1][k];
              B[k+Ncoef*2][j*COORD_DIM+0]=M0[j*6+2][k];
              B[k+Ncoef*0][j*COORD_DIM+1]=M0[j*6+1][k];
              B[k+Ncoef*1][j*COORD_DIM+1]=M0[j*6+3][k];
              B[k+Ncoef*2][j*COORD_DIM+1]=M0[j*6+4][k];
              B[k+Ncoef*0][j*COORD_DIM+2]=M0[j*6+2][k];
              B[k+Ncoef*1][j*COORD_DIM+2]=M0[j*6+4][k];
              B[k+Ncoef*2][j*COORD_DIM+2]=M0[j*6+5][k];
            }
          }
          Matrix<Real> M1(Ncoef*COORD_DIM, COORD_DIM*Ngrid, DL3.begin()+i*COORD_DIM*Ncoef*COORD_DIM*Ngrid, false);
          for(Long k=0;k<B.Dim(0);k++){ // Rearrange
            for(Long j0=0;j0<COORD_DIM;j0++){
              for(Long j1=0;j1<p0+1;j1++){
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+   j1)*2+0)*p0+j2]=B[k][((j1*p0+j2)*2+0)*COORD_DIM+j0];
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+p0-j1)*2+1)*p0+j2]=B[k][((j1*p0+j2)*2+1)*COORD_DIM+j0];
              }
            }
          }
        }
      }
    }
  }
  Profile::Toc();


  Profile::Tic("Grid2SHC");
  SphericalHarmonics<Real>::Grid2SHC(SL3, p0+1, 2*p0, p0, SL, SHCArrange::COL_MAJOR_NONZERO);
  SphericalHarmonics<Real>::Grid2SHC(DL3, p0+1, 2*p0, p0, DL, SHCArrange::COL_MAJOR_NONZERO);
  Profile::Toc();

}

}  // end namespace
