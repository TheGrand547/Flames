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

// From here https://github.com/chrischristakis/Winston-Shield/blob/master/shaders/bubble.fs
static const std::string linearDepth = "float LinearizeDepth(float depth) {float zNdc = 2 * depth - 1;float zEye = (2 * far * near) / ((far + near) - zNdc * (far - near));float linearDepth = (zEye - near) / (far - near);return linearDepth;}";

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

// From here https://stackoverflow.com/questions/5281261/generating-a-normal-map-from-a-height-map
static const std::string heightToNormalFragment = "#version 460 core\n"
"in vec2 uv;"
"out vec4 normal;"
"const ivec3 off = ivec3(-1, 0, 1);"
"uniform sampler2D heightMap;"
"void main()"
"{"
"	const vec2 size = fwidthFine(uv);"
"	float s11 = texture(heightMap, uv).r;"
"	float s01 = textureOffset(heightMap, uv, off.xy).r;"
"	float s21 = textureOffset(heightMap, uv, off.zy).r;"
"	float s10 = textureOffset(heightMap, uv, off.yx).r;"
"	float s12 = textureOffset(heightMap, uv, off.yz).r;"
"   float xDelta = s21 - s01;"
//"   float partialAX = s21 - s11; float partialBX = s11 - s01;"
//"   xDelta = (abs(partialAX) < abs(partialBX)) ? partialAX : partialBX;"
//"   float partialAY = s12 - s11; float partialBY = s11 - s10;"
//"   xDelta = min(s21 - s11, s11 - s01);"
//"   if (sign(s21 - s11) != sign(s11 - s01)) xDelta = 0;"
"   float yDelta = s12 - s10;"
//"   yDelta = (abs(partialAY) < abs(partialBY)) ? partialAY : partialBY;"
//"   yDelta = min(s12 - s11, s11 - s10);"
//"   if (sign(s12 - s11) != sign(s11 - s10)) yDelta = 0;"

"	vec3 va = normalize(vec3(size.x, xDelta, size.y));"
"   vec3 vb = normalize(vec3(size.y, yDelta, -size.x));"
"	normal = vec4(normalize(cross(va, vb)) / 2 + 0.5, 1);"
"}";


void HeightToNormal(const Texture2D& input, Texture2D& output)
{
	output.CreateEmpty(input.GetSize(), InternalRGBA16);
	ColorFrameBuffer buffer;
	buffer.GetColor().MakeAliasOf(output);
	buffer.Assemble();
	buffer.Bind();
	input.SetViewport();

	Shader temp;
	temp.CompileEmbedded(identityVertex, heightToNormalFragment);

	DisableGLFeatures<Blending>();
	temp.SetActiveShader();
	temp.SetTextureUnit("heightMap", input);
	temp.DrawArray<DrawType::TriangleStrip>(4);
	BindDefaultFrameBuffer();
}

// Stuff for Voronoi Noise

