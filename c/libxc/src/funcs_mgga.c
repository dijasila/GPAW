#include "util.h"

extern XC(func_info_type) XC(func_info_mgga_x_tpss);
extern XC(func_info_type) XC(func_info_mgga_c_tpss);
extern XC(func_info_type) XC(func_info_mgga_x_m06l);
extern XC(func_info_type) XC(func_info_mgga_c_m06l);
extern XC(func_info_type) XC(func_info_mgga_x_revtpss);
extern XC(func_info_type) XC(func_info_mgga_c_revtpss);
extern XC(func_info_type) XC(func_info_mgga_x_otpss);
extern XC(func_info_type) XC(func_info_mgga_c_otpss);
extern XC(func_info_type) XC(func_info_mgga_x_mbeef);
extern XC(func_info_type) XC(func_info_mgga_x_ms0);
extern XC(func_info_type) XC(func_info_mgga_x_ms1);
extern XC(func_info_type) XC(func_info_mgga_x_ms2);


const XC(func_info_type) *XC(mgga_known_funct)[] = {
  &XC(func_info_mgga_x_tpss),
  &XC(func_info_mgga_c_tpss),
  &XC(func_info_mgga_x_m06l),
  &XC(func_info_mgga_c_m06l),
  &XC(func_info_mgga_x_revtpss),
  &XC(func_info_mgga_c_revtpss),
  &XC(func_info_mgga_x_otpss),
  &XC(func_info_mgga_c_otpss),
  &XC(func_info_mgga_x_mbeef),
  &XC(func_info_mgga_x_ms0),
  &XC(func_info_mgga_x_ms1),
  &XC(func_info_mgga_x_ms2),
  NULL
};
