// Header of static arrays of monomial coeffs of spreading kernel function in each
// fine-grid interval. Generated by gen_all_horner_cpp_header.m in finufft/devel
// Authors: Alex Barnett, Ludvig af Klinteberg, Marco Barbone & Libin Lu.
// (C) 2018--2024 The Simons Foundation, Inc.
#include <array>

template<uint8_t w> constexpr auto nc125() noexcept {
  constexpr uint8_t ncs[] = {4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 13, 14, };
  return ncs[w-2];
}

template<class T, uint8_t w>
constexpr std::array<std::array<T, w>, nc125<w>()> get_horner_coeffs_125() noexcept {
    constexpr auto nc = nc125<w>();
    if constexpr (w==2) {
      return std::array<std::array<T, w>, nc> {{
          {-1.9075708590566734E-01, 1.9075708590566728E-01},
          {-9.0411309581634888E-02, -9.0411309581634625E-02},
          {6.4742429432896442E-01, -6.4742429432896453E-01},
          {6.1209111871385724E-01, 6.1209111871385691E-01},
      }};
    } else if constexpr (w==3) {
      return std::array<std::array<T, w>, nc> {{
          {-2.9270010751775016E-02, 3.7966707032750742E-02, -2.9270010728701203E-02},
          {-4.4469294619149641E-02, -3.1573886092308317E-18, 4.4469294640111512E-02},
          {1.4864411342268646E-01, -3.0473448739822762E-01, 1.4864411344492173E-01},
          {4.0470611346184532E-01, -4.2425842671824539E-17, -4.0470611343822149E-01},
          {2.4728112933307078E-01, 1.0000000000000044E+00, 2.4728112935494950E-01},
      }};
    } else if constexpr (w==4) {
      return std::array<std::array<T, w>, nc> {{
          {-3.0464394190490465E-03, 5.3247889205097444E-03, -5.3247889205097912E-03, 3.0464394190490543E-03},
          {-1.0401300825285587E-02, 6.3725646657140341E-03, 6.3725646657139994E-03, -1.0401300825285620E-02},
          {1.5677587697716076E-02, -6.7022293289915644E-02, 6.7022293289915699E-02, -1.5677587697716051E-02},
          {1.1425598262146333E-01, -1.1126112046907125E-01, -1.1126112046907125E-01, 1.1425598262146337E-01},
          {1.7431588385887239E-01, 3.7425489538028406E-01, -3.7425489538028428E-01, -1.7431588385887237E-01},
          {8.4048892491849825E-02, 7.9275732207620908E-01, 7.9275732207620886E-01, 8.4048892491849783E-02},
      }};
    } else if constexpr (w==5) {
      return std::array<std::array<T, w>, nc> {{
          {-1.5212353034889752E-03, 1.7151925122365424E-03, 8.6737859434123232E-18, -1.7151925122365892E-03, 1.5212353034889810E-03},
          {-2.3306908700109395E-05, -8.3858973028988343E-03, 1.4886952481383780E-02, -8.3858973028988516E-03, -2.3306908700108859E-05},
          {1.8780973157032109E-02, -3.8322611720715674E-02, 2.0186098450281545E-17, 3.8322611720715723E-02, -1.8780973157032105E-02},
          {5.4855980576944546E-02, 3.7709308632020662E-02, -1.8284069243892614E-01, 3.7709308632020801E-02, 5.4855980576944560E-02},
          {6.2936773057387069E-02, 3.7198919402374026E-01, -8.4851685343650335E-17, -3.7198919402374014E-01, -6.2936773057387083E-02},
          {2.5811126752233304E-02, 4.6616226852477355E-01, 1.0000000000000004E+00, 4.6616226852477299E-01, 2.5811126752233314E-02},
      }};
    } else if constexpr (w==6) {
      return std::array<std::array<T, w>, nc> {{
          {-1.6161497824910401E-04, 2.5924418389359724E-04, -1.3917099193206471E-04, -1.3917099193200356E-04, 2.5924418389357561E-04, -1.6161497812640463E-04},
          {-2.6471424081647520E-04, -5.6150758897068509E-04, 2.0099203466670676E-03, -2.0099203466671066E-03, 5.6150758897069290E-04, 2.6471424094083525E-04},
          {1.9378826192716983E-03, -6.8365127179467666E-03, 4.7406536657959653E-03, 4.7406536657959306E-03, -6.8365127179467709E-03, 1.9378826194070362E-03},
          {1.0463664645794023E-02, -5.8671703446042042E-03, -3.4019677093840475E-02, 3.4019677093840461E-02, 5.8671703446041964E-03, -1.0463664645671077E-02},
          {2.1435449512033442E-02, 7.4190333865239919E-02, -9.5369600014193201E-02, -9.5369600014193159E-02, 7.4190333865239960E-02, 2.1435449512163873E-02},
          {2.0397684222696253E-02, 2.4277466601214734E-01, 2.6509440217151270E-01, -2.6509440217151259E-01, -2.4277466601214734E-01, -2.0397684222557694E-02},
          {7.3992041846532757E-03, 2.2998056434514016E-01, 8.5775196559356104E-01, 8.5775196559356104E-01, 2.2998056434514022E-01, 7.3992041847816149E-03},
      }};
    } else if constexpr (w==7) {
      return std::array<std::array<T, w>, nc> {{
          {-1.2555096177147631E-05, 2.7293834771950587E-05, -2.6660039700443368E-05, -1.5789308062419179E-17, 2.6660039700526404E-05, -2.7293834771958475E-05, 1.2555096209062448E-05},
          {-4.7399003259806255E-05, 2.0950491942900027E-06, 1.7484854214666628E-04, -2.9104069274775099E-04, 1.7484854214662188E-04, 2.0950491942971965E-06, -4.7399003227280481E-05},
          {1.1260116639581571E-04, -7.8814564904711420E-04, 1.1036556706848707E-03, -1.7239739988593079E-17, -1.1036556706849018E-03, 7.8814564904712190E-04, -1.1260116636284838E-04},
          {1.3572774007773840E-03, -2.3954706749181320E-03, -2.9058644824981553E-03, 7.8619155407045668E-03, -2.9058644824981211E-03, -2.3954706749181446E-03, 1.3572774008132624E-03},
          {4.4924606632387679E-03, 7.2245566707420791E-03, -2.7743312484355673E-02, -5.9183130302535822E-17, 2.7743312484355704E-02, -7.2245566707420964E-03, -4.4924606632061829E-03},
          {7.4065234100227726E-03, 5.7825030729344355E-02, 1.0889852837591876E-04, -1.3060049459923273E-01, 1.0889852837587447E-04, 5.7825030729344383E-02, 7.4065234100573743E-03},
          {6.1353661835569220E-03, 1.2822551681002711E-01, 3.1973557271594355E-01, -6.3638764007737718E-17, -3.1973557271594361E-01, -1.2822551681002708E-01, -6.1353661835202144E-03},
          {2.0163149398992283E-03, 1.0071602557045134E-01, 5.8653557849806126E-01, 1.0000000000000004E+00, 5.8653557849806159E-01, 1.0071602557045131E-01, 2.0163149399332566E-03},
      }};
    } else if constexpr (w==8) {
      return std::array<std::array<T, w>, nc> {{
          {-5.1256159860521059E-06, 5.3292178505827560E-06, 8.7427989025241135E-06, -2.8404799465034010E-05, 2.8404799465103762E-05, -8.7427989025340797E-06, -5.3292178505848321E-06, 5.1256159860515867E-06},
          {-6.5665874015875498E-07, -6.1884865695206945E-05, 1.4476791315359098E-04, -8.6782118193358648E-05, -8.6782118193387420E-05, 1.4476791315357120E-04, -6.1884865695208748E-05, -6.5665874015841775E-07},
          {1.2504911757628661E-04, -3.9351755557265425E-04, 2.3739384784470553E-05, 9.6592347103021444E-04, -9.6592347103024556E-04, -2.3739384784447222E-05, 3.9351755557266206E-04, -1.2504911757628680E-04},
          {6.5470478006265378E-04, 5.7029826102786423E-05, -4.0842122325117809E-03, 3.3746160664395804E-03, 3.3746160664395288E-03, -4.0842122325117896E-03, 5.7029826102776760E-05, 6.5470478006265378E-04},
          {1.6676293877589655E-03, 8.1606118103203715E-03, -1.0603838868224464E-02, -2.0559571166483818E-02, 2.0559571166483891E-02, 1.0603838868224413E-02, -8.1606118103203749E-03, -1.6676293877589668E-03},
          {2.3525728171808302E-03, 3.3585505340219680E-02, 4.4733940386002188E-02, -8.0668262921248610E-02, -8.0668262921248568E-02, 4.4733940386002209E-02, 3.3585505340219701E-02, 2.3525728171808302E-03},
          {1.7458301875074103E-03, 5.9145446836664561E-02, 2.5435204236257858E-01, 2.0538938722823227E-01, -2.0538938722823244E-01, -2.5435204236257863E-01, -5.9145446836664561E-02, -1.7458301875074092E-03},
          {5.2827275612461451E-04, 4.0402734444109238E-02, 3.4389230803369708E-01, 8.9161099745784866E-01, 8.9161099745784866E-01, 3.4389230803369697E-01, 4.0402734444109224E-02, 5.2827275612461430E-04},
      }};
    } else if constexpr (w==9) {
      return std::array<std::array<T, w>, nc> {{
          {-4.1446092652960729E-07, 7.2790527337694249E-07, -2.5130319770320465E-08, -1.9002349620399918E-06, 3.0493470976490761E-06, -1.9002349619053566E-06, -2.5130319758080882E-08, 7.2790527337216141E-07, -4.1446092652953556E-07},
          {-9.3772917893937511E-07, -3.0575635011733582E-06, 1.2977675432522508E-05, -1.5241881422240646E-05, 9.0538532849680740E-18, 1.5241881422370184E-05, -1.2977675432515243E-05, 3.0575635011791710E-06, 9.3772917893919352E-07},
          {7.9668965137352613E-06, -4.2137454928179153E-05, 3.9856859670067282E-05, 6.5639620808994340E-05, -1.4477186949848808E-04, 6.5639620808769517E-05, 3.9856859670051101E-05, -4.2137454928182751E-05, 7.9668965137352613E-06},
          {7.0565721004957926E-05, -9.0876125855048783E-05, -3.5965836571495268E-04, 7.0575785995722685E-04, 1.3852933204327093E-17, -7.0575785995747578E-04, 3.5965836571492157E-04, 9.0876125855048783E-05, -7.0565721004957953E-05},
          {2.6358216867957524E-04, 7.0803132065997234E-04, -2.3883045659485323E-03, -1.0047843626592211E-03, 4.8455486978740345E-03, -1.0047843626590149E-03, -2.3883045659485410E-03, 7.0803132065997180E-04, 2.6358216867957530E-04},
          {5.6197289626769558E-04, 5.4583505067802903E-03, 8.8722695781043813E-04, -2.0386313118366247E-02, -2.5586386887323148E-17, 2.0386313118366479E-02, -8.8722695781043510E-04, -5.4583505067803007E-03, -5.6197289626769590E-04},
          {7.0217948741779833E-04, 1.6533012331430411E-02, 4.8637875368588449E-02, -1.5084170630532941E-02, -1.0157816246607008E-01, -1.5084170630533198E-02, 4.8637875368588469E-02, 1.6533012331430449E-02, 7.0217948741779833E-04},
          {4.7572953640583418E-04, 2.4761567630011038E-02, 1.6332247709293546E-01, 2.7616213278983209E-01, -6.3638764007737718E-17, -2.7616213278983237E-01, -1.6332247709293557E-01, -2.4761567630011107E-02, -4.7572953640583407E-04},
          {1.3409415535124442E-04, 1.5141199617983766E-02, 1.8004032483820079E-01, 6.6268423293859668E-01, 1.0000000000000004E+00, 6.6268423293859735E-01, 1.8004032483820079E-01, 1.5141199617983816E-02, 1.3409415535124450E-04},
      }};
    } else if constexpr (w==10) {
      return std::array<std::array<T, w>, nc> {{
          {-2.5033479260931243E-08, 6.3042298325822974E-08, -5.2303271559903752E-08, -7.6226091793981470E-08, 2.3316553106919865E-07, -2.3316553114670068E-07, 7.6226091852107994E-08, 5.2303271559903752E-08, -6.3042298325822974E-08, 2.5033479260954218E-08},
          {-1.2089379439830882E-07, -3.4743143854854381E-08, 8.2889801007304441E-07, -1.5830293785166221E-06, 8.7461219396318958E-07, 8.7461219390199166E-07, -1.5830293786451377E-06, 8.2889801007763427E-07, -3.4743143856671194E-08, -1.2089379439833272E-07},
          {2.7742147829396434E-07, -3.2550081973300800E-06, 5.9212960378052008E-06, 8.5495977199931152E-07, -1.3248468528067400E-05, 1.3248468528201922E-05, -8.5495977192042543E-07, -5.9212960378031248E-06, 3.2550081973319486E-06, -2.7742147829393835E-07},
          {5.7780052154064923E-06, -1.5636835808662441E-05, -1.6121807313048698E-05, 8.1230533420432682E-05, -5.5456530742837516E-05, -5.5456530742837516E-05, 8.1230533420447061E-05, -1.6121807313052293E-05, -1.5636835808664237E-05, 5.7780052154064500E-06},
          {3.1258610702677757E-05, 2.8169545035124896E-05, -2.9881406711975588E-04, 1.5956798534240196E-04, 5.3653099874330845E-04, -5.3653099874345622E-04, -1.5956798534228530E-04, 2.9881406711975979E-04, -2.8169545035121007E-05, -3.1258610702677757E-05},
          {9.4710355505531879E-05, 6.0621452710061727E-04, -7.0118560592789044E-04, -2.4750745659638880E-03, 2.4757076628502080E-03, 2.4757076628501222E-03, -2.4750745659639995E-03, -7.0118560592789261E-04, 6.0621452710061011E-04, 9.4710355505531744E-05},
          {1.7649923565147223E-04, 2.9221990881931068E-03, 4.9086823797164910E-03, -1.0940556313145935E-02, -1.3762152424114771E-02, 1.3762152424114837E-02, 1.0940556313145999E-02, -4.9086823797165006E-03, -2.9221990881930985E-03, -1.7649923565147191E-04},
          {1.9968216068682140E-04, 7.2783782301876591E-03, 3.5949398124193947E-02, 2.5847993600195466E-02, -6.9275634160640545E-02, -6.9275634160640365E-02, 2.5847993600195553E-02, 3.5949398124193933E-02, 7.2783782301876418E-03, 1.9968216068682094E-04},
          {1.2517797191066984E-04, 9.6269418565961429E-03, 9.1130577457178424E-02, 2.4769645835465365E-01, 1.6766875916810517E-01, -1.6766875916810545E-01, -2.4769645835465348E-01, -9.1130577457178438E-02, -9.6269418565961117E-03, -1.2517797191066959E-04},
          {3.3157481538170295E-05, 5.3715860775974443E-03, 8.6328042282845754E-02, 4.3077092326437988E-01, 9.1242439930731112E-01, 9.1242439930731112E-01, 4.3077092326437960E-01, 8.6328042282845754E-02, 5.3715860775974201E-03, 3.3157481538170274E-05},
      }};
    } else if constexpr (w==11) {
      return std::array<std::array<T, w>, nc> {{
          {-9.8984999695150684E-09, 1.0194946774237274E-08, 3.5279000674744131E-08, -1.1638771467098967E-07, 1.2326133614997474E-07, 4.0516498100293627E-17, -1.2326133616658230E-07, 1.1638771461009523E-07, -3.5279000675436117E-08, -1.0194946774540017E-08, 9.8984999695035805E-09},
          {-7.7890073973121893E-09, -1.8340559948721496E-07, 5.4451797328899156E-07, -3.5830285714158528E-07, -7.3873233539148814E-07, 1.4648976903565062E-06, -7.3873233539148814E-07, -3.5830285714158528E-07, 5.4451797329166893E-07, -1.8340559948707155E-07, -7.7890073973062137E-09},
          {3.5729663467784788E-07, -1.6085054296210825E-06, 4.5672370507419953E-07, 6.0608527683396937E-06, -9.0233724844419958E-06, 4.9941702496811109E-18, 9.0233724844635849E-06, -6.0608527683001474E-06, -4.5672370507336912E-07, 1.6085054296209787E-06, -3.5729663467787059E-07},
          {2.8888404081262420E-06, -1.8976367884802011E-06, -2.4767547607262255E-05, 3.8337725458138998E-05, 2.6462355617083014E-05, -8.2113719362889533E-05, 2.6462355617074022E-05, 3.8337725458128211E-05, -2.4767547607268550E-05, -1.8976367884808755E-06, 2.8888404081262352E-06},
          {1.2057435171015772E-05, 4.6687328398363199E-05, -1.3963494372748247E-04, -1.4877651674415632E-04, 4.6954815721688515E-04, -8.0112266480761140E-18, -4.6954815721699401E-04, 1.4877651674410965E-04, 1.3963494372748247E-04, -4.6687328398363565E-05, -1.2057435171015740E-05},
          {3.1396100602888598E-05, 3.6443237253636128E-04, 1.5906780001786376E-04, -1.9495384184342525E-03, -2.4621376046556591E-04, 3.2818730060400242E-03, -2.4621376046549717E-04, -1.9495384184342957E-03, 1.5906780001786054E-04, 3.6443237253636090E-04, 3.1396100602888476E-05},
          {5.2504054888010103E-05, 1.3660648269306120E-03, 4.7357572177382573E-03, -2.2373255422689078E-03, -1.5459233729560838E-02, 8.2777271348319417E-17, 1.5459233729560940E-02, 2.2373255422689585E-03, -4.7357572177382573E-03, -1.3660648269306120E-03, -5.2504054888009899E-05},
          {5.4612928019025150E-05, 2.9497743530118282E-03, 2.1858479505161198E-02, 3.8333708936616487E-02, -2.1641923687039311E-02, -8.3109405654057139E-02, -2.1641923687039148E-02, 3.8333708936616549E-02, 2.1858479505161187E-02, 2.9497743530118282E-03, 5.4612928019024878E-05},
          {3.1996260415636094E-05, 3.5282769389657653E-03, 4.5889527487056471E-02, 1.8012194355267486E-01, 2.4178022040260394E-01, -1.4849044935138820E-16, -2.4178022040260408E-01, -1.8012194355267491E-01, -4.5889527487056485E-02, -3.5282769389657670E-03, -3.1996260415635877E-05},
          {8.0191950887587672E-06, 1.8211144887695901E-03, 3.8565497751765716E-02, 2.5236459439543668E-01, 7.1517256669690443E-01, 1.0000000000000000E+00, 7.1517256669690399E-01, 2.5236459439543651E-01, 3.8565497751765709E-02, 1.8211144887695910E-03, 8.0191950887586656E-06},
      }};
    } else if constexpr (w==12) {
      return std::array<std::array<T, w>, nc> {{
          {-6.3791929313927889E-10, 1.2240176129392066E-09, 5.3586929655729871E-10, -6.2807356207213370E-09, 1.0600657345063913E-08, -5.5585209137321311E-09, -5.5585209024191334E-09, 1.0600657351506037E-08, -6.2807355779833450E-09, 5.3586929058654981E-10, 1.2240176130570502E-09, -6.3791928984364409E-10},
          {-2.0816585198663231E-09, -6.8192670392721950E-09, 3.6338774646281261E-08, -4.9464521005206807E-08, -1.3242031043825771E-08, 1.0671664853011416E-07, -1.0671664860484826E-07, 1.3242031029986123E-08, 4.9464520965071825E-08, -3.6338774641091391E-08, 6.8192670390235131E-09, 2.0816585232936298E-09},
          {1.5395324498811026E-08, -1.2022118042098672E-07, 1.5464523856461759E-07, 2.7605497715117822E-07, -8.4964626030792186E-07, 5.2067203455623376E-07, 5.2067203460519205E-07, -8.4964626028956253E-07, 2.7605497715882793E-07, 1.5464523855945402E-07, -1.2022118042095684E-07, 1.5395324502815186E-08},
          {2.1206307767330490E-07, -4.5869687934425177E-07, -1.3462277877572238E-06, 4.2970047520095079E-06, -1.1214870287414941E-06, -6.9831974682611276E-06, 6.9831974682960042E-06, 1.1214870288062637E-06, -4.2970047519858427E-06, 1.3462277877584693E-06, 4.5869687934430365E-07, -2.1206307766916437E-07},
          {1.2088615636792283E-06, 2.2204932634070340E-06, -1.5559909809164321E-05, 1.8771595438438357E-06, 4.7304527720930092E-05, -3.7055029721542363E-05, -3.7055029721474024E-05, 4.7304527720924698E-05, 1.8771595438150590E-06, -1.5559909809162522E-05, 2.2204932634069218E-06, 1.2088615636834516E-06},
          {4.2345162286123920E-06, 3.3664241555334188E-05, -3.0535096226552359E-05, -1.9795772057291372E-04, 1.7526295499601351E-04, 3.2830037656729569E-04, -3.2830037656741235E-04, -1.7526295499597461E-04, 1.9795772057291762E-04, 3.0535096226554304E-05, -3.3664241555334127E-05, -4.2345162286081289E-06},
          {9.7900673700200710E-06, 1.8351475200221906E-04, 3.8725987583789449E-04, -9.2229408802589803E-04, -1.5383560041741977E-03, 1.8800996948122673E-03, 1.8800996948122159E-03, -1.5383560041741806E-03, -9.2229408802591950E-04, 3.8725987583788858E-04, 1.8351475200221892E-04, 9.7900673700247601E-06},
          {1.4953735432776068E-05, 5.8049865432805055E-04, 3.2684769908807644E-03, 2.3619245295514002E-03, -1.0074268581043128E-02, -9.8551520939613012E-03, 9.8551520939613984E-03, 1.0074268581043190E-02, -2.3619245295513390E-03, -3.2684769908807631E-03, -5.8049865432805055E-04, -1.4953735432771904E-05},
          {1.4462226804444718E-05, 1.1205076408888253E-03, 1.1698445222077601E-02, 3.3958877046121605E-02, 1.3705098421608818E-02, -6.0497400607811502E-02, -6.0497400607811364E-02, 1.3705098421608861E-02, 3.3958877046121584E-02, 1.1698445222077629E-02, 1.1205076408888253E-03, 1.4462226804449268E-05},
          {7.9801239249145906E-06, 1.2318344820958854E-03, 2.1335987794357202E-02, 1.1394981969310450E-01, 2.3520579283187470E-01, 1.4166451219687684E-01, -1.4166451219687690E-01, -2.3520579283187470E-01, -1.1394981969310465E-01, -2.1335987794357230E-02, -1.2318344820958849E-03, -7.9801239249098591E-06},
          {1.9028495068410013E-06, 5.9416527261081902E-04, 1.6248140264385584E-02, 1.3597036436097915E-01, 4.9821957378204829E-01, 9.2652305802242962E-01, 9.2652305802242918E-01, 4.9821957378204829E-01, 1.3597036436097931E-01, 1.6248140264385619E-02, 5.9416527261081913E-04, 1.9028495068454209E-06},
      }};
    } else if constexpr (w==13) {
      return std::array<std::array<T, w>, nc> {{
          {-2.9813639428880222E-11, 8.8416967203798553E-11, -6.1944900049416202E-11, -2.3424446304741974E-10, 6.6123634170186695E-10, -6.5395826611340155E-10, -4.1544196918380560E-17, 6.5395798843212787E-10, -6.6123634887104090E-10, 2.3424448292235756E-10, 6.1944899375087956E-11, -8.8416967518780826E-11, 2.9813639428672270E-11},
          {-1.9473100882661174E-10, -6.0076128537522075E-11, 1.8131864336846848E-09, -3.9994904594475383E-09, 2.0334605453276710E-09, 5.0274131644549980E-09, -9.3367591454043182E-09, 5.0274135399836745E-09, 2.0334605013326797E-09, -3.9994904896155324E-09, 1.8131864336846848E-09, -6.0076128203631512E-11, -1.9473100882576781E-10},
          {2.7912946705499348E-10, -6.8584366112252110E-09, 1.5876438439921649E-08, 2.2894800333296164E-09, -5.4355139618053459E-08, 6.9215572172708388E-08, -2.0761222756661531E-17, -6.9215572327712453E-08, 5.4355139609749671E-08, -2.2894800250258273E-09, -1.5876438439921649E-08, 6.8584366109549049E-09, -2.7912946705575371E-10},
          {1.2350515865275413E-08, -4.7668301905193214E-08, -3.2637845350503965E-08, 3.2101904613886198E-07, -3.3650826993121675E-07, -3.1117289067535446E-07, 7.8771611533367254E-07, -3.1117289078551072E-07, -3.3650826986389902E-07, 3.2101904612203255E-07, -3.2637845350169294E-08, -4.7668301904846586E-08, 1.2350515865276535E-08},
          {9.7956192761409367E-08, 9.2080334894896161E-09, -1.2031586234331795E-06, 1.3860784486026173E-06, 2.8079238803139698E-06, -5.6034103145542327E-06, -1.2739341879231602E-17, 5.6034103145492504E-06, -2.8079238803461473E-06, -1.3860784486009564E-06, 1.2031586234342174E-06, -9.2080334897361341E-09, -9.7956192761410584E-08},
          {4.5216719173889329E-07, 2.3203195635244806E-06, -6.0547210914040705E-06, -1.2111482379350871E-05, 3.0238388566349212E-05, 1.0632529352104186E-05, -5.0954659549722760E-05, 1.0632529352261560E-05, 3.0238388566306048E-05, -1.2111482379354468E-05, -6.0547210914051945E-06, 2.3203195635247335E-06, 4.5216719173889398E-07},
          {1.3873038503072794E-06, 1.8694798962849904E-05, 1.4885937076477319E-05, -1.3109520271107600E-04, -4.6797213058836700E-05, 3.2555441892430831E-04, -4.2789977886166419E-17, -3.2555441892428500E-04, 4.6797213058883368E-05, 1.3109520271106627E-04, -1.4885937076476833E-05, -1.8694798962849965E-05, -1.3873038503072805E-06},
          {2.9080869014384416E-06, 8.2405696428180866E-05, 3.3386109283452882E-04, -1.7130036080580300E-04, -1.5108662980936798E-03, 7.8665018928628880E-05, 2.3686576883603285E-03, 7.8665018928680434E-05, -1.5108662980936711E-03, -1.7130036080580300E-04, 3.3386109283452703E-04, 8.2405696428180690E-05, 2.9080869014384429E-06},
          {4.1089519307370109E-06, 2.2941839162878702E-04, 1.8941440042457411E-03, 3.5673079836347600E-03, -3.6880489041049127E-03, -1.2074156718545229E-02, 7.7638840193391308E-18, 1.2074156718545398E-02, 3.6880489041048680E-03, -3.5673079836347752E-03, -1.8941440042457402E-03, -2.2941839162878607E-04, -4.1089519307370134E-06},
          {3.7267581324409626E-06, 4.0381251792508718E-04, 5.7019503038218408E-03, 2.4040868593456798E-02, 2.9406233528281676E-02, -2.4394921635639274E-02, -7.0323343245740924E-02, -2.4394921635639076E-02, 2.9406233528281728E-02, 2.4040868593456794E-02, 5.7019503038218391E-03, 4.0381251792508517E-04, 3.7267581324409639E-06},
          {1.9487148068106048E-06, 4.1285069961250690E-04, 9.2995630713278779E-03, 6.5021145064983535E-02, 1.8663042875530000E-01, 2.1451870821533811E-01, -4.2425842671825291E-17, -2.1451870821533800E-01, -1.8663042875529998E-01, -6.5021145064983424E-02, -9.2995630713278762E-03, -4.1285069961250441E-04, -1.9487148068106044E-06},
          {4.4408051211162671E-07, 1.8756193861873413E-04, 6.5146989208011699E-03, 6.8352802598867848E-02, 3.1564238810082496E-01, 7.5353649746793927E-01, 9.9999999999999944E-01, 7.5353649746793827E-01, 3.1564238810082473E-01, 6.8352802598867710E-02, 6.5146989208011664E-03, 1.8756193861873272E-04, 4.4408051211162740E-07},
      }};
    } else if constexpr (w==14) {
      return std::array<std::array<T, w>, nc> {{
          {-1.4791529084475183E-12, 4.8147158230661728E-12, -7.1247156948137733E-12, -3.7363541207502799E-12, 3.0923963981839685E-11, -4.7998355799044319E-11, 2.4268806716162679E-11, 2.4268733214440741E-11, -4.7998325173326849E-11, 3.0923986440699169E-11, -3.7363622876082732E-12, -7.1247172580014364E-12, 4.8147157243700130E-12, -1.4791527913704449E-12},
          {-1.2240623323565210E-11, 1.4269095045857491E-11, 6.3689195678622977E-11, -2.3523039312409202E-10, 2.6546832599550726E-10, 9.4137124835858634E-11, -5.6473808447746174E-10, 5.6473802485264837E-10, -9.4137192268683287E-11, -2.6546836370465266E-10, 2.3523038893615868E-10, -6.3689194596148682E-11, -1.4269094972657386E-11, 1.2240623457626768E-11},
          {-2.3785683828648566E-11, -2.9453404128779668E-10, 1.0997757892316608E-09, -8.6020469050217690E-10, -2.2974593148638725E-09, 5.5064436862062262E-09, -3.1470906039204597E-09, -3.1470906435159520E-09, 5.5064436312124872E-09, -2.2974593224058709E-09, -8.6020468893092726E-10, 1.0997757877782549E-09, -2.9453404132462281E-10, -2.3785683688841006E-11},
          {5.5138523621058489E-10, -3.4792607432848048E-09, 2.1621109683219443E-09, 1.6802313214377317E-08, -3.4440501477287078E-08, 3.6408052006210212E-09, 5.4274262219974878E-08, -5.4274262276717443E-08, -3.6408052283003184E-09, 3.4440501466215358E-08, -1.6802313213339344E-08, -2.1621109680192019E-09, 3.4792607432685862E-09, -5.5138523606432426E-10},
          {6.5041263396090684E-09, -9.9149367808892008E-09, -6.6845758889566122E-08, 1.6286641993591861E-07, 5.8507874937330077E-08, -4.7688540979254475E-07, 3.2559878513865341E-07, 3.2559878505297634E-07, -4.7688540973134683E-07, 5.8507875016887355E-08, 1.6286641992979879E-07, -6.6845758890092050E-08, -9.9149367809190818E-09, 6.5041263397797216E-09},
          {3.9336515129721029E-08, 1.1257285216172823E-07, -6.2406181937643518E-07, -2.6873173855212429E-07, 2.8292088258391734E-06, -1.4598715517943753E-06, -4.0212462690573659E-06, 4.0212462690777108E-06, 1.4598715517561777E-06, -2.8292088259366911E-06, 2.6873173855004832E-07, 6.2406181937633131E-07, -1.1257285216170229E-07, -3.9336515129545111E-08},
          {1.5611302559652624E-07, 1.4859455506706588E-06, -8.5826557923801041E-07, -1.1616353402589941E-05, 8.0333594878743668E-06, 2.8616079443342253E-05, -2.5816776957596172E-05, -2.5816776957725667E-05, 2.8616079443297517E-05, 8.0333594878743668E-06, -1.1616353402589941E-05, -8.5826557923812285E-07, 1.4859455506706096E-06, 1.5611302559670729E-07},
          {4.3045614796951609E-07, 8.9716871724550223E-06, 2.3377513570381975E-05, -5.5213296993544491E-05, -1.2391624765754029E-04, 1.5869855385558889E-04, 2.1530382494139264E-04, -2.1530382494148987E-04, -1.5869855385559667E-04, 1.2391624765754420E-04, 5.5213296993542547E-05, -2.3377513570382097E-05, -8.9716871724550562E-06, -4.3045614796933784E-07},
          {8.3014334976692694E-07, 3.4045323043173907E-05, 2.1660980714121266E-04, 1.7421792587401537E-04, -9.2118064021565096E-04, -9.7597008655070415E-04, 1.4714477548414425E-03, 1.4714477548413221E-03, -9.7597008655066978E-04, -9.2118064021557355E-04, 1.7421792587402128E-04, 2.1660980714121377E-04, 3.4045323043173954E-05, 8.3014334976713256E-07},
          {1.0954436997682012E-06, 8.5568590196649072E-05, 9.7778250562911362E-04, 3.0692948752812726E-03, 6.0463237460737715E-04, -8.9532302111319517E-03, -7.4040784665310088E-03, 7.4040784665311398E-03, 8.9532302111319049E-03, -6.0463237460742030E-04, -3.0692948752812760E-03, -9.7778250562911796E-04, -8.5568590196649234E-05, -1.0954436997680322E-06},
          {9.3810713124204517E-07, 1.3926941499858522E-04, 2.5833386162538992E-03, 1.4797516242328845E-02, 3.0361769467151939E-02, 5.7261067343619453E-03, -5.3608938764866568E-02, -5.3608938764866804E-02, 5.7261067343620043E-03, 3.0361769467151887E-02, 1.4797516242328843E-02, 2.5833386162539074E-03, 1.3926941499858538E-04, 9.3810713124224771E-07},
          {4.6718564624239798E-07, 1.3360375098030156E-04, 3.8410346178215297E-03, 3.4207779106833432E-02, 1.2923501383683486E-01, 2.2132894130184300E-01, 1.2264779624530257E-01, -1.2264779624530266E-01, -2.2132894130184300E-01, -1.2923501383683503E-01, -3.4207779106833432E-02, -3.8410346178215393E-03, -1.3360375098030183E-04, -4.6718564624220306E-07},
          {1.0213002307223056E-07, 5.7528591418445632E-05, 2.5031206020280083E-03, 3.2405046511689239E-02, 1.8485678142025511E-01, 5.5177865704975293E-01, 9.3670793123951734E-01, 9.3670793123951734E-01, 5.5177865704975293E-01, 1.8485678142025552E-01, 3.2405046511689246E-02, 2.5031206020280179E-03, 5.7528591418445801E-05, 1.0213002307242283E-07},
      }};
    } else if constexpr (w==15) {
      return std::array<std::array<T, w>, nc> {{
          {-6.7275763610714952E-13, 1.4037883799551298E-12, 1.0122748703359079E-12, -1.0507011558415453E-11, 1.9186622029950655E-11, -7.9757821000147658E-12, -2.2999231890282221E-11, 4.0853143156922655E-11, -2.2999289058288172E-11, -7.9759209366006470E-12, 1.9186575580945818E-11, -1.0507009389093798E-11, 1.0122747666550936E-12, 1.4037883779612681E-12, -6.7275763607599545E-13},
          {-2.9809392328870459E-12, -8.3268200106445154E-12, 5.7687950421418410E-11, -9.1929199328063136E-11, -3.9289939147449750E-11, 3.0713723883726984E-10, -3.5332678542514975E-10, -2.7146437259190364E-19, 3.5332673644762448E-10, -3.0713734467131358E-10, 3.9289960193589238E-11, 9.1929195849949024E-11, -5.7687950527891289E-11, 8.3268200078717850E-12, 2.9809392328263928E-12},
          {1.6106092880560131E-11, -1.9612809867391534E-10, 3.3667881343327413E-10, 5.4740705721569177E-10, -2.3219918274241966E-09, 1.8783264065861090E-09, 2.1531914801939135E-09, -4.8374639374556914E-09, 2.1531914732804149E-09, 1.8783263688761161E-09, -2.3219918425081937E-09, 5.4740705894406642E-10, 3.3667881359039910E-10, -1.9612809867391534E-10, 1.6106092880566125E-11},
          {3.5447644664515603E-10, -1.1390658479631380E-09, -2.4324028601744043E-09, 1.2152005525649129E-08, -7.1102518397187493E-09, -2.5878341881540945E-08, 4.0855407208672653E-08, -5.2935793078520102E-17, -4.0855407200368865E-08, 2.5878341983954342E-08, 7.1102518798537297E-09, -1.2152005534644900E-08, 2.4324028599797840E-09, 1.1390658479621243E-09, -3.5447644664522991E-10},
          {2.8405432421064880E-09, 2.6648052024110037E-09, -4.5328290134669312E-08, 3.2089634827933056E-08, 1.7241593347433905E-07, -2.5816631649430664E-07, -1.3664009514955699E-07, 4.6017883229632535E-07, -1.3664009519851534E-07, -2.5816631659222327E-07, 1.7241593343456043E-07, 3.2089634833287873E-08, -4.5328290134639428E-08, 2.6648052024095099E-09, 2.8405432421065231E-09},
          {1.4373673262756718E-08, 9.2554419735720107E-08, -2.0417866965620877E-07, -6.8820764686365081E-07, 1.4165168644119495E-06, 1.2531774951342180E-06, -3.6383191328395853E-06, 7.7231484391689009E-17, 3.6383191329101677E-06, -1.2531774953177319E-06, -1.4165168644026078E-06, 6.8820764686022552E-07, 2.0417866965633852E-07, -9.2554419735716864E-08, -1.4373673262756819E-08},
          {5.0693377499403671E-08, 7.7594237801399261E-07, 9.4933483676650543E-07, -6.6987818302441095E-06, -4.5889941143310731E-06, 2.2647907184666643E-05, 3.7412856035719476E-06, -3.3754692339531092E-05, 3.7412856035755445E-06, 2.2647907184632474E-05, -4.5889941143094898E-06, -6.6987818302360161E-06, 9.4933483676673021E-07, 7.7594237801399261E-07, 5.0693377499403691E-08},
          {1.2779800356186583E-07, 3.8997040140349321E-06, 1.8264189394307380E-05, -8.3632912035133083E-06, -1.0687544349165045E-04, 2.2123224044259885E-06, 2.3404180714519442E-04, -3.1780101629791230E-17, -2.3404180714510888E-04, -2.2123224042859913E-06, 1.0687544349166601E-04, 8.3632912035016430E-06, -1.8264189394307563E-05, -3.8997040140349321E-06, -1.2779800356186591E-07},
          {2.2919642176438707E-07, 1.3183839322480003E-05, 1.2030953406839345E-04, 2.4905754342428545E-04, -3.4193403196993528E-04, -1.1551611179404116E-03, 2.1954335627570980E-04, 1.7895433812202518E-03, 2.1954335627569262E-04, -1.1551611179404632E-03, -3.4193403196995035E-04, 2.4905754342428572E-04, 1.2030953406839334E-04, 1.3183839322479994E-05, 2.2919642176438718E-07},
          {2.8457821671573232E-07, 3.0427184404092272E-05, 4.6337319534911801E-04, 2.1072304367244893E-03, 2.4342755210407336E-03, -4.2814200474568642E-03, -9.6703299158782570E-03, 1.1713135756253251E-16, 9.6703299158783472E-03, 4.2814200474569067E-03, -2.4342755210407171E-03, -2.1072304367244859E-03, -4.6337319534911801E-04, -3.0427184404092272E-05, -2.8457821671573253E-07},
          {2.3137327105312781E-07, 4.6266060425611184E-05, 1.1028009511991968E-03, 8.2352859806754768E-03, 2.4233386066663400E-02, 2.2182889945939446E-02, -2.5327411650384966E-02, -6.0946897479642374E-02, -2.5327411650384900E-02, 2.2182889945939362E-02, 2.4233386066663431E-02, 8.2352859806754889E-03, 1.1028009511991974E-03, 4.6266060425611204E-05, 2.3137327105312789E-07},
          {1.1019919454791575E-07, 4.1938159428224133E-05, 1.5154850601194975E-03, 1.6839357628952684E-02, 8.0835952724673241E-02, 1.8739074372244102E-01, 1.9255567517255726E-01, 1.6956773054418529E-31, -1.9255567517255731E-01, -1.8739074372244113E-01, -8.0835952724673366E-02, -1.6839357628952709E-02, -1.5154850601194975E-03, -4.1938159428224133E-05, -1.1019919454791579E-07},
          {2.3183302143948740E-08, 1.7202745817468638E-05, 9.2668857465754892E-04, 1.4607490553401940E-02, 1.0130044556641118E-01, 3.7041488405244688E-01, 7.8279781886019206E-01, 1.0000000000000011E+00, 7.8279781886019195E-01, 3.7041488405244721E-01, 1.0130044556641132E-01, 1.4607490553401953E-02, 9.2668857465754838E-04, 1.7202745817468631E-05, 2.3183302143948743E-08},
      }};
    } else if constexpr (w==16) {
      return std::array<std::array<T, w>, nc> {{
          {-2.3198933270740715E-14, 8.4680084926105710E-14, -5.5120394376955525E-14, -3.4224825412770884E-13, 1.0093451766215381E-12, -9.9669407015123244E-13, -4.1950464449360547E-13, 2.1120456723238401E-12, -2.1120631281451168E-12, 4.1949988381507543E-13, 9.9669407015123244E-13, -1.0093463667911707E-12, 3.4224835330851150E-13, 5.5120381979355185E-14, -8.4680084587108829E-14, 2.3198933258633685E-14},
          {-2.1496737417083317E-13, -2.2214974042200800E-14, 2.3291735717266144E-12, -5.9732917765233235E-12, 3.0556712628179253E-12, 1.1858122635605482E-11, -2.4316415414833160E-11, 1.3235499986994189E-11, 1.3235536737855158E-11, -2.4316433790263641E-11, 1.1858112427032992E-11, 3.0556697315320517E-12, -5.9732921593447914E-12, 2.3291735677388905E-12, -2.2214973792968073E-14, -2.1496737416207108E-13},
          {1.7715918240672815E-14, -8.7094275514577869E-12, 2.5402078534858863E-11, 5.6643120203537577E-13, -1.1273397749808333E-10, 1.7831198930961025E-10, 2.2123190757406476E-13, -2.7985827080469500E-10, 2.7985821912985675E-10, -2.2124766556045869E-13, -1.7831199578671051E-10, 1.1273397565255340E-10, -5.6643233774610691E-13, -2.5402078583658933E-11, 8.7094275509032414E-12, -1.7715918237423519E-14},
          {1.5548426850867747E-11, -8.2967690041035768E-11, -2.0776280275005410E-11, 6.5818716252940090E-10, -9.7473366764093964E-10, -7.2114134421445299E-10, 2.9974008586911667E-09, -1.8729407766830212E-09, -1.8729408099935145E-09, 2.9974008676571101E-09, -7.2114133321570515E-10, -9.7473366606969001E-10, 6.5818716284365085E-10, -2.0776280294646031E-11, -8.2967690039501348E-11, 1.5548426850871941E-11},
          {1.7210848751139206E-10, -1.3819378018485677E-10, -2.4707116695746685E-09, 4.6626394244300632E-09, 6.2513494738369478E-09, -2.2225751670676472E-08, 7.2716681748129466E-09, 2.9914504847745951E-08, -2.9914504925247984E-08, -7.2716681969563846E-09, 2.2225751655452858E-08, -6.2513494779888428E-09, -4.6626394252950410E-09, 2.4707116695043892E-09, 1.3819378018409654E-10, -1.7210848751141845E-10},
          {1.1055703983904745E-09, 4.3691209554203762E-09, -2.0201061499410946E-08, -2.3275033897663606E-08, 1.2633562931562412E-07, -2.2021804055570054E-08, -2.7912172398560873E-07, 2.1280289566371563E-07, 2.1280289566371563E-07, -2.7912172399172855E-07, -2.2021804070869530E-08, 1.2633562931791905E-07, -2.3275033897807039E-08, -2.0201061499398992E-08, 4.3691209554207493E-09, 1.1055703983904803E-09},
          {4.8970459380161164E-09, 5.4304148291616929E-08, -1.0066736763230802E-08, -5.3239387743771126E-07, 2.2987809872367560E-07, 1.8048974519479544E-06, -1.3449315565629853E-06, -2.4760016204856650E-06, 2.4760016205163890E-06, 1.3449315566227727E-06, -1.8048974519269872E-06, -2.2987809871931610E-07, 5.3239387743823018E-07, 1.0066736763256750E-08, -5.4304148291616929E-08, -4.8970459380161164E-09},
          {1.5672684443241270E-08, 3.5812571134853209E-07, 1.1292168823202786E-06, -2.5215449854178345E-06, -7.6275609266392180E-06, 9.3973092319735694E-06, 1.7891569285079721E-05, -1.8642776809377737E-05, -1.8642776809426295E-05, 1.7891569285078386E-05, 9.3973092319897562E-06, -7.6275609266437149E-06, -2.5215449854180594E-06, 1.1292168823201941E-06, 3.5812571134853119E-07, 1.5672684443241263E-08},
          {3.6571939291734580E-08, 1.5742222553115409E-06, 1.1217451065775842E-05, 1.0668471374318627E-05, -6.0694020243069901E-05, -7.4268888177613095E-05, 1.3567546096380107E-04, 1.4875477215032178E-04, -1.4875477215047734E-04, -1.3567546096381663E-04, 7.4268888177613095E-05, 6.0694020243062122E-05, -1.0668471374319355E-05, -1.1217451065775811E-05, -1.5742222553115415E-06, -3.6571939291734593E-08},
          {6.1501023800531295E-08, 4.8443034242391141E-06, 6.0167136036954503E-05, 2.0573318254802077E-04, 1.2811955521419976E-05, -8.3782209201438288E-04, -6.2669687707128208E-04, 1.1809008871738671E-03, 1.1809008871739529E-03, -6.2669687707129921E-04, -8.3782209201441725E-04, 1.2811955521421050E-05, 2.0573318254802050E-04, 6.0167136036954456E-05, 4.8443034242391132E-06, 6.1501023800531308E-08},
          {7.2283166867263303E-08, 1.0391634193778160E-05, 2.0529674430143854E-04, 1.2618687081127932E-03, 2.6256301814801060E-03, -5.5040645592551222E-04, -7.8709464111364341E-03, -5.7657980103486698E-03, 5.7657980103487626E-03, 7.8709464111365313E-03, 5.5040645592551373E-04, -2.6256301814801060E-03, -1.2618687081127928E-03, -2.0529674430143856E-04, -1.0391634193778164E-05, -7.2283166867263303E-08},
          {5.6049296769722387E-08, 1.4879146623074258E-05, 4.4787865139353365E-04, 4.2383440773521696E-03, 1.6624620601556193E-02, 2.6395394769117640E-02, 3.6740117889106082E-04, -4.8088574473126630E-02, -4.8088574473126713E-02, 3.6740117889113471E-04, 2.6395394769117678E-02, 1.6624620601556193E-02, 4.2383440773521722E-03, 4.4787865139353359E-04, 1.4879146623074262E-05, 5.6049296769722367E-08},
          {2.5620581163903708E-08, 1.2815874111792787E-05, 5.7471335914300670E-04, 7.8386860177525539E-03, 4.6638901641906962E-02, 1.3897554029141571E-01, 2.0773808644544137E-01, 1.0813440420918320E-01, -1.0813440420918335E-01, -2.0773808644544145E-01, -1.3897554029141571E-01, -4.6638901641906975E-02, -7.8386860177525539E-03, -5.7471335914300648E-04, -1.2815874111792787E-05, -2.5620581163903721E-08},
          {5.2012152104083984E-09, 5.0291159580938677E-06, 3.3201112337137920E-04, 6.3015433246683353E-03, 5.2427915343763412E-02, 2.3104762006593377E-01, 5.9521037322997217E-01, 9.4441119081353875E-01, 9.4441119081353875E-01, 5.9521037322997217E-01, 2.3104762006593377E-01, 5.2427915343763412E-02, 6.3015433246683353E-03, 3.3201112337137920E-04, 5.0291159580938694E-06, 5.2012152104083868E-09},
      }};
    } else {
      static_assert(w >= 2, "w must be >= 2");
      static_assert(w <= 16, "w must be <= 16");
      return {};
    }
};
