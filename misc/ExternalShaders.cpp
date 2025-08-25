#include "ExternalShaders.h"
#include "../Shader.h"

namespace ExternalShaders
{
	void Setup()
	{
		constexpr auto externals = std::to_array<std::string_view>(
			{
				"CellularNoise",
				"debug",
				"fbm",
				"fbmImage",
				"gradientNoise",
				"hash",
				"hexagons",
				"interpolate",
				"metric",
				"multiHash",
				"noise",
				"patterns",
				"perlinNoise",
				"voronoi",
				"warp"
			}
		);
		for (const auto& element : externals)
		{
			Shader::IncludeInShaderFilesystem(element.data(), std::string("external\\") + element.data() + std::string(".glsl"));
		}
		Shader::IncludeInShaderFilesystem("CubeMapMath", "CubeMapMath.incl");
	}
}