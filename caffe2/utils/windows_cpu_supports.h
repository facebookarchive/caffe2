#if defined(_MSC_VER)
#include <intrin.h>	
inline bool __builtin_cpu_supports(const char * avx2)
{																					
	bool avx2Supported = false;														
	int cpuInfo[4];																	
	__cpuid(cpuInfo, 1);															
	bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;						
	bool cpuAVX2Suport = cpuInfo[2] & (1 << 28) || false;							
	if (osUsesXSAVE_XRSTORE && cpuAVX2Suport)										
	{																				
		unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);	
		avx2Supported = (xcrFeatureMask & 0x6) == 0x6;								
	}																				
	return avx2Supported;															
}
#endif
