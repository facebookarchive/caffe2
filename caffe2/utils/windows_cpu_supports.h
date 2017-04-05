#if defined(_MSC_VER)
#include <intrin.h>	
inline bool __builtin_cpu_supports(const char * avx2)
{																					
	bool avxSupported = false;														
	int cpuInfo[4];																	
	__cpuid(cpuInfo, 1);															
	bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;						
	bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;							
	if (osUsesXSAVE_XRSTORE && cpuAVXSuport)										
	{																				
		unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);	
		avxSupported = (xcrFeatureMask & 0x6) == 0x6;								
	}																				
	return avxSupported;															
}
#endif
