#ifndef __CUFINUFFT_ERRORS_H__
#define __CUFINUFFT_ERRORS_H__

// For error checking
static const char* _cufftGetErrorEnum(cufftResult_t error)
{
	switch(error)
	{
		case CUFFT_SUCCESS:
			return "cufft_success";
		case CUFFT_INVALID_PLAN:
			return "cufft_invalid_plan";
		case CUFFT_ALLOC_FAILED:
			return "cufft_alloc_failed";
		case CUFFT_INVALID_TYPE:
			return "cufft_invalid_type";
		case CUFFT_INVALID_VALUE:
			return "cufft_invalid_value";
		case CUFFT_INTERNAL_ERROR:
			return "cufft_internal_error";
		case CUFFT_EXEC_FAILED:
			return "cufft_exec_failed";
		case CUFFT_SETUP_FAILED:
			return "cufft_setup_failed";
		case CUFFT_INVALID_SIZE:
			return "cufft_invalid_size";
		case CUFFT_UNALIGNED_DATA:
			return "cufft_unaligned data";
		case CUFFT_INCOMPLETE_PARAMETER_LIST:
			return "cufft_incomplete_parameter_list";
		case CUFFT_INVALID_DEVICE:
			return "cufft_invalid_device";
		case CUFFT_PARSE_ERROR:
			return "cufft_parse_error";
		case CUFFT_NO_WORKSPACE:
			return "cufft_no_workspace";
		case CUFFT_NOT_IMPLEMENTED:
			return "cufft_not_implemented";
		case CUFFT_LICENSE_ERROR:
			return "cufft_license_error";
		case CUFFT_NOT_SUPPORTED:
			return "cufft_not_supported";
	}
	return "<unknown>";
}

#endif
