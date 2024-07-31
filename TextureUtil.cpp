#include "TextureUtil.h"
#include "glmHelp.h"
#include "glUtil.h"
#include "Framebuffer.h"
#include "Shader.h"

// TODO: Setup shaders for here
/*
20x20 tiles(for the moment) 


*/

// Takes the desired size
std::array<ScreenRect, 9> NineSliceGenerate(glm::ivec2 topLeft, glm::ivec2& size)
{
	const glm::ivec2 tileSize(20, 20);
	size = glm::max(size, 2 * tileSize);
	const glm::ivec2 clipped = size - 2 * tileSize;
	const glm::ivec2 edge = size - tileSize;
	// Assume size is bigger than the minimum 40x40
	std::array<ScreenRect, 9> rects =
	{
		{
			{glm::vec4(0, 0, tileSize)}, 
			{glm::vec4(tileSize.x, 0, clipped.x, tileSize.y)},
			{glm::vec4(edge.x, 0, tileSize)},
			{glm::vec4(0, tileSize.y, tileSize.x, clipped.y)},
			{glm::vec4(tileSize, clipped)},
			{glm::vec4(edge.x, tileSize.y, tileSize.x, clipped.y)},
			{glm::vec4(0, edge.y, tileSize)},
			{glm::vec4(tileSize.x, edge.y, clipped.x, tileSize.y)},
			{glm::vec4(edge, tileSize)}
		}
	};
	for (auto& ref : rects)
	{
		ref += glm::vec4(topLeft, 0, 0);
	}
	return rects;
}


static const std::string identityVertex = "#version 440 core\nout vec2 uv;vec2 positions[] = {\tvec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),\tvec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)};vec2 uvCoords[] = {\tvec2(0.0f, 0.0f), vec2(1.0f, 0.0f), \tvec2(0.0f, 1.0f), vec2(1.0f, 1.0f)};void main(){\tgl_Position = vec4(positions[gl_VertexID].xy, 0, 1);\tuv = uvCoords[gl_VertexID];}";

static const std::string voronoiFragment =
"#version 440 core"
"layout(location = 0) in vec2 uv"
"layout(std140) uniform Points"
"{"
"int length;"
"vec2 points[32];"
"};"
"void main()"
"{"
"for (int i = 0; i < Points.length; i++)"
"{"
"for (int "
"}"
"";

static const std::string heightToNormalFragment = "#version 440 core\n"
"in vec2 uv;"
"out vec4 normal;"
"const ivec3 off = ivec3(-1, 0, 1);"
"uniform sampler2D heightMap;"
"void main()"
"{"
"	vec2 size = fwidth(uv) / 2;"
"	vec4 wave = texture(heightMap, uv);"
"	float s11 = wave.r;"
"	float s01 = textureOffset(heightMap, uv, off.xy).r;"
"	float s21 = textureOffset(heightMap, uv, off.zy).r;"
"	float s10 = textureOffset(heightMap, uv, off.yx).r;"
"	float s12 = textureOffset(heightMap, uv, off.yz).r;"

"	vec3 va = normalize(vec3(size.xy, s21 - s01));"
"	vec3 vb = normalize(vec3(size.yx, s12 - s10));"
"	normal = vec4(cross(va, vb), s11);"
"}";


void HeightToNormal(const Texture2D& input, Texture2D& output)
{
	output.CreateEmpty(input.GetSize());
	ColorFrameBuffer buffer;
	buffer.GetColor().MakeAliasOf(output);
	buffer.Assemble();
	buffer.Bind();

	Shader temp;
	temp.CompileEmbedded(identityVertex, heightToNormalFragment);

	DisableGLFeatures<Blending>();
	temp.SetActiveShader();
	temp.SetTextureUnit("heightMap", input);
	temp.DrawArray<DrawType::TriangleStrip>(4);
	BindDefaultFrameBuffer();
}

// Stuff for Voronoi Noise

