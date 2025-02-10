#include "Parallel.h"

// PARALLEL MODE RUNDOWN
// std::execution::seq       - Standard sequential execution of the operation
// std::execution::par       - Standard parallel execution of the operation
// std::execution::unseq     - Vectorized execution of the operation(SIMD type stuff)
// std::execution::par_unseq - Indicates the operation can be executed in any way, parallel, vectorized, etc

namespace Parallel
{
	// TODO: Maybe mutex this or something idk
	bool IsEnabled = true;

	bool Enabled() noexcept
	{
		return ::Parallel::IsEnabled;
	}


	void SetStatus(bool status) noexcept
	{
		::Parallel::IsEnabled = status;
	}
}
