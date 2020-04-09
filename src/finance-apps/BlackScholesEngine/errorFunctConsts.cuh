// errorFunctConsts.cuh
// Scott Grauer-Gray
// April 29, 2012
// Constants for the error function

#ifndef ERROR_FUNCT_CONSTS_CUH
#define ERROR_FUNCT_CONSTS_CUH

#define DBL_MIN (1e-999)

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

#ifndef M_2_PI
#define M_2_PI 0.636619772367581343076
#endif

#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244008443621048490392848359376887
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244008443621048490392848359376887
#endif

#define ERROR_FUNCT_tiny 0.000000000000000000001f // QL_EPSILON
#define ERROR_FUNCT_one 1.00000000000000000000e+00
/* c = (float)0.84506291151 */
#define ERROR_FUNCT_erx 8.45062911510467529297e-01 /* 0x3FEB0AC1, 0x60000000   \
                                                    */
                                                   //
// Coefficients for approximation to  erf on [0,0.84375]
//
#define ERROR_FUNCT_efx 1.28379167095512586316e-01  /* 0x3FC06EBA, 0x8214DB69  \
                                                     */
#define ERROR_FUNCT_efx8 1.02703333676410069053e+00 /* 0x3FF06EBA, 0x8214DB69  \
                                                     */
#define ERROR_FUNCT_pp0 1.28379167095512558561e-01  /* 0x3FC06EBA, 0x8214DB68  \
                                                     */
#define ERROR_FUNCT_pp1 -3.25042107247001499370e-01 /* 0xBFD4CD7D, 0x691CB913  \
                                                     */
#define ERROR_FUNCT_pp2 -2.84817495755985104766e-02 /* 0xBF9D2A51, 0xDBD7194F  \
                                                     */
#define ERROR_FUNCT_pp3 -5.77027029648944159157e-03 /* 0xBF77A291, 0x236668E4  \
                                                     */
#define ERROR_FUNCT_pp4 -2.37630166566501626084e-05 /* 0xBEF8EAD6, 0x120016AC  \
                                                     */
#define ERROR_FUNCT_qq1 3.97917223959155352819e-01  /* 0x3FD97779, 0xCDDADC09  \
                                                     */
#define ERROR_FUNCT_qq2 6.50222499887672944485e-02  /* 0x3FB0A54C, 0x5536CEBA  \
                                                     */
#define ERROR_FUNCT_qq3 5.08130628187576562776e-03  /* 0x3F74D022, 0xC4D36B0F  \
                                                     */
#define ERROR_FUNCT_qq4 1.32494738004321644526e-04  /* 0x3F215DC9, 0x221C1A10  \
                                                     */
#define ERROR_FUNCT_qq5 -3.96022827877536812320e-06 /* 0xBED09C43, 0x42A26120  \
                                                     */
                                                    //
// Coefficients for approximation to  erf  in [0.84375,1.25]
//
#define ERROR_FUNCT_pa0 -2.36211856075265944077e-03 /* 0xBF6359B8, 0xBEF77538  \
                                                     */
#define ERROR_FUNCT_pa1 4.14856118683748331666e-01  /* 0x3FDA8D00, 0xAD92B34D  \
                                                     */
#define ERROR_FUNCT_pa2 -3.72207876035701323847e-01 /* 0xBFD7D240, 0xFBB8C3F1  \
                                                     */
#define ERROR_FUNCT_pa3 3.18346619901161753674e-01  /* 0x3FD45FCA, 0x805120E4  \
                                                     */
#define ERROR_FUNCT_pa4 -1.10894694282396677476e-01 /* 0xBFBC6398, 0x3D3E28EC  \
                                                     */
#define ERROR_FUNCT_pa5 3.54783043256182359371e-02  /* 0x3FA22A36, 0x599795EB  \
                                                     */
#define ERROR_FUNCT_pa6 -2.16637559486879084300e-03 /* 0xBF61BF38, 0x0A96073F  \
                                                     */
#define ERROR_FUNCT_qa1 1.06420880400844228286e-01  /* 0x3FBB3E66, 0x18EEE323  \
                                                     */
#define ERROR_FUNCT_qa2 5.40397917702171048937e-01  /* 0x3FE14AF0, 0x92EB6F33  \
                                                     */
#define ERROR_FUNCT_qa3 7.18286544141962662868e-02  /* 0x3FB2635C, 0xD99FE9A7  \
                                                     */
#define ERROR_FUNCT_qa4 1.26171219808761642112e-01  /* 0x3FC02660, 0xE763351F  \
                                                     */
#define ERROR_FUNCT_qa5 1.36370839120290507362e-02  /* 0x3F8BEDC2, 0x6B51DD1C  \
                                                     */
#define ERROR_FUNCT_qa6 1.19844998467991074170e-02  /* 0x3F888B54, 0x5735151D  \
                                                     */
//
// Coefficients for approximation to  erfc in [1.25,1/0.35]
//
#define ERROR_FUNCT_ra0 -9.86494403484714822705e-03 /* 0xBF843412, 0x600D6435  \
                                                     */
#define ERROR_FUNCT_ra1 -6.93858572707181764372e-01 /* 0xBFE63416, 0xE4BA7360  \
                                                     */
#define ERROR_FUNCT_ra2 -1.05586262253232909814e+01 /* 0xC0251E04, 0x41B0E726  \
                                                     */
#define ERROR_FUNCT_ra3 -6.23753324503260060396e+01 /* 0xC04F300A, 0xE4CBA38D  \
                                                     */
#define ERROR_FUNCT_ra4 -1.62396669462573470355e+02 /* 0xC0644CB1, 0x84282266  \
                                                     */
#define ERROR_FUNCT_ra5 -1.84605092906711035994e+02 /* 0xC067135C, 0xEBCCABB2  \
                                                     */
#define ERROR_FUNCT_ra6 -8.12874355063065934246e+01 /* 0xC0545265, 0x57E4D2F2  \
                                                     */
#define ERROR_FUNCT_ra7 -9.81432934416914548592e+00 /* 0xC023A0EF, 0xC69AC25C  \
                                                     */
#define ERROR_FUNCT_sa1 1.96512716674392571292e+01  /* 0x4033A6B9, 0xBD707687  \
                                                     */
#define ERROR_FUNCT_sa2 1.37657754143519042600e+02  /* 0x4061350C, 0x526AE721  \
                                                     */
#define ERROR_FUNCT_sa3 4.34565877475229228821e+02  /* 0x407B290D, 0xD58A1A71  \
                                                     */
#define ERROR_FUNCT_sa4 6.45387271733267880336e+02  /* 0x40842B19, 0x21EC2868  \
                                                     */
#define ERROR_FUNCT_sa5 4.29008140027567833386e+02  /* 0x407AD021, 0x57700314  \
                                                     */
#define ERROR_FUNCT_sa6 1.08635005541779435134e+02  /* 0x405B28A3, 0xEE48AE2C  \
                                                     */
#define ERROR_FUNCT_sa7 6.57024977031928170135e+00  /* 0x401A47EF, 0x8E484A93  \
                                                     */
#define ERROR_FUNCT_sa8 -6.04244152148580987438e-02 /* 0xBFAEEFF2, 0xEE749A62  \
                                                     */
//
// Coefficients for approximation to  erfc in [1/.35,28]
//
#define ERROR_FUNCT_rb0 -9.86494292470009928597e-03 /* 0xBF843412, 0x39E86F4A  \
                                                     */
#define ERROR_FUNCT_rb1 -7.99283237680523006574e-01 /* 0xBFE993BA, 0x70C285DE  \
                                                     */
#define ERROR_FUNCT_rb2 -1.77579549177547519889e+01 /* 0xC031C209, 0x555F995A  \
                                                     */
#define ERROR_FUNCT_rb3 -1.60636384855821916062e+02 /* 0xC064145D, 0x43C5ED98  \
                                                     */
#define ERROR_FUNCT_rb4 -6.37566443368389627722e+02 /* 0xC083EC88, 0x1375F228  \
                                                     */
#define ERROR_FUNCT_rb5 -1.02509513161107724954e+03 /* 0xC0900461, 0x6A2E5992  \
                                                     */
#define ERROR_FUNCT_rb6 -4.83519191608651397019e+02 /* 0xC07E384E, 0x9BDC383F  \
                                                     */
#define ERROR_FUNCT_sb1 3.03380607434824582924e+01  /* 0x403E568B, 0x261D5190  \
                                                     */
#define ERROR_FUNCT_sb2 3.25792512996573918826e+02  /* 0x40745CAE, 0x221B9F0A  \
                                                     */
#define ERROR_FUNCT_sb3 1.53672958608443695994e+03  /* 0x409802EB, 0x189D5118  \
                                                     */
#define ERROR_FUNCT_sb4 3.19985821950859553908e+03  /* 0x40A8FFB7, 0x688C246A  \
                                                     */
#define ERROR_FUNCT_sb5 2.55305040643316442583e+03  /* 0x40A3F219, 0xCEDF3BE6  \
                                                     */
#define ERROR_FUNCT_sb6 4.74528541206955367215e+02  /* 0x407DA874, 0xE79FE763  \
                                                     */
#define ERROR_FUNCT_sb7 -2.24409524465858183362e+01 /* 0xC03670E2, 0x42712D62  \
                                                     */

#endif // ERROR_FUNCT_CONSTS_CUH
