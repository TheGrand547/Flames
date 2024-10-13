#include "Level.h"

namespace Level
{
	void Clear() noexcept
	{
		Geometry.Clear();
		AllNodes.clear();
		Tree.Clear();
	}
}
