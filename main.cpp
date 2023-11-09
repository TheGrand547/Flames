#include <algorithm>
#include <chrono>
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/hash.hpp>
#include <freeglut.h>
#include <iostream>
#include <map>
#include <queue>
#include <sys/utime.h>
#include <time.h>
#include <unordered_map>
#include "AABB.h"
#include "Buffer.h"
#include "CubeMap.h"
#include "Font.h"
#include "Framebuffer.h"
#include "glmHelp.h"
#include "Lines.h"
#include "log.h"
#include "Model.h"
#include "OrientedBoundingBox.h"
#include "Plane.h"
#include "Shader.h"
#include "Sphere.h"
#include "StaticOctTree.h"
#include "stbWrangler.h"
#include "Texture2D.h"
#include "UniformBuffer.h"
#include "util.h"
#include "Vertex.h"
#include "VertexArray.h"
#include "Wall.h"

struct Dummy
{
	OBB box;
	bool color;
};

template <class T> inline void CombineVector(std::vector<T>& left, const std::vector<T>& right)
{
	left.insert(left.end(), std::make_move_iterator(right.begin()), std::make_move_iterator(right.end()));
}

// TODO: GRRRRR DO THIS BETTER DUMB DUMB FUCKER
std::array<ColoredVertex, 8> coloredCubeVertex{
	{
		{{-1, -1, -1}, {1, 1, 1}},
		{{ 1, -1, -1}, {0, 1, 1}},
		{{ 1,  1, -1}, {0, 0, 1}},
		{{-1,  1, -1}, {1, 0, 1}},
		{{-1, -1,  1}, {1, 1, 0}},
		{{ 1, -1,  1}, {0, 1, 0}},
		{{ 1,  1,  1}, {0, 0, 0}},
		{{-1,  1,  1}, {1, 0, 0}},
	}
};

static const std::array<glm::vec3, 8> plainCubeVerts {
	{
		{-1, -1, -1},   // -x, -y, -z
		{ 1, -1, -1},   // +x, -y, -z
		{ 1,  1, -1},   // +x, +y, -z
		{-1,  1, -1},   // -x, +y, -z
		{-1, -1,  1},   // -x, -y, +z
		{ 1, -1,  1},   // +x, -y, +z
		{ 1,  1,  1},   // +x, +y, +z
		{-1,  1,  1},   // -x, +y, +z
	}
};

// TODO: Better indexing so vertiex texture coordinates don't have to be repeated with in the same face
// If j = (index) % 6, then j = 0/4 are unique, j = 1/2 are repeated as 3/5 respectively
static const std::array<GLubyte, 36> cubeIndicies =
{
	0, 4, 3, // -X Face
	4, 7, 3,

	0, 1, 4, // -Y Face
	1, 5, 4,

	1, 0, 2, // -Z Face
	0, 3, 2,

	6, 5, 2, // +X Face
	5, 1, 2,

	6, 2, 7, // +Y Face
	2, 3, 7,

	7, 4, 6, // +Z Face
	4, 5, 6,
};
// I don't know what's goign on with this but I didn't like the old thing

std::array<glm::vec3, cubeIndicies.size()> texturedCubeVerts =
	[](auto verts, auto index) constexpr
	{
		std::array<glm::vec3, index.size()> temp{};
		for (int i = 0; i < temp.size(); i++)
		{
			temp[i] = verts[index[i]];
		}
		return temp;
	} (plainCubeVerts, cubeIndicies);

std::array<GLubyte, 24> cubeOutline =
{
	0, 1,  1, 2,  2, 3,  3, 0, 
	4, 5,  5, 6,  6, 7,  7, 4, 
	2, 6,  5, 1, 
	3, 7,  4, 0, 
};

// EW
std::array<Vertex, 4> plane{
	{
		{ 1, 0,  1},
		{ 1, 0, -1},
		{-1, 0,  1},
		{-1, 0, -1}
	}
};

std::array<glm::vec3, 10> stick{
	{
		{0,   0, -.5},
		{0,   0,  .5},
		{0, .85,   0},
		{0, 1.7,   0},
		{0, 1.5,   0},
		{0, 1.2, -.5},
		{0, 1.2,  .5},
		{0, 1.9,   0},
		{0, 1.8, -.2},
		{0, 1.8,  .2},
	}
};

std::array<GLubyte, 14> stickDex = { 0, 2, 1, 2, 4, 5, 4, 6, 4, 3, 8, 7, 9, 3 };

std::array<GLubyte, 5> planeOutline = { 0, 1, 3, 2, 0 }; // Can be done with one less vertex using GL_LINE_LOOP

static const std::array<GLubyte, 16 * 16> dither16 = {
{
	0,   191,  48, 239,  12, 203,  60, 251,   3, 194,  51, 242,  15, 206,  63, 254,
	127,  64, 175, 112, 139,  76, 187, 124, 130,  67, 178, 115, 142,  79, 190, 127,
	 32, 223,  16, 207,  44, 235,  28, 219,  35, 226,  19, 210,  47, 238,  31, 222,
	159,  96, 143,  80, 171, 108, 155,  92, 162,  99, 146,  83, 174, 111, 158,  95,
	  8, 199,  56, 247,   4, 195,  52, 243,  11, 202,  59, 250,   7, 198,  55, 246,
	135,  72, 183, 120, 131,  68, 179, 116, 138,  75, 186, 123, 134,  71, 182, 119,
	 40, 231,  24, 215,  36, 227,  20, 211,  43, 234,  27, 218,  39, 230,  23, 214,
	167, 104, 151,  88, 163, 100, 147,  84, 170, 107, 154,  91, 166, 103, 150,  87,
	  2, 193,  50, 241,  14, 205,  62, 253,   1, 192,  49, 240,  13, 204,  61, 252,
	129,  66, 177, 114, 141,  78, 189, 126, 128,  65, 176, 113, 140,  77, 188, 125,
	 34, 225,  18, 209,  46, 237,  30, 221,  33, 224,  17, 208,  45, 236,  29, 220,
	161,  98, 145,  82, 173, 110, 157,  94, 160,  97, 144,  81, 172, 109, 156,  93,
	 10, 201,  58, 249,   6, 197,  54, 245,   9, 200,  57, 248,   5, 196,  53, 244,
	137,  74, 185, 122, 133,  70, 181, 118, 136,  73, 184, 121, 132,  69, 180, 117,
	 42, 233,  26, 217,  38, 229,  22, 213,  41, 232,  25, 216,  37, 228,  21, 212,
	169, 106, 153,  90, 165, 102, 149,  86, 168, 105, 152,  89, 164, 101, 148,  85
} };

const int ditherSize = 16;

// Buffers
Buffer<ArrayBuffer> albertBuffer, capsuleBuffer, instanceBuffer, plainCube, planeBO, rayBuffer, sphereBuffer, stickBuffer, texturedPlane;
Buffer<ElementArray> capsuleIndex, cubeOutlineIndex, sphereIndicies, stickIndicies;

UniformBuffer cameraUniformBuffer, screenSpaceBuffer;

// Framebuffer
Framebuffer<2, Depth> depthed;
ColorFrameBuffer scratchSpace;

// Shaders
Shader dither, expand, finalResult, frameShader, flatLighting, instancing, uiRect, uiRectTexture, uniform, sphereMesh, widget;

// Textures
Texture2D ditherTexture, hatching, texture, wallTexture;
CubeMap mapper;

// Vertex Array Objects
VAO instanceVAO, meshVAO, normalVAO, plainVAO, texturedVAO;


// TODO: Make structures around these
GLuint framebufferMod;


// Not explicitly tied to OpenGL Globals
OBB smartBox, dumbBox;
std::vector<Model> planes;
StaticOctTree<Dummy> boxes(glm::vec3(20));

static unsigned int frameCounter = 0;
bool smartBoxColor = false;

glm::vec3 moveSphere(0, 3.5f, 6.5f);
int kernel = 0;
int lineWidth = 3;

#define TIGHT_BOXES 1
#define WIDE_BOXES 2
// One for each number key
std::array<bool, '9' - '0' + 1> debugFlags{};


// Input Shenanigans
#define ArrowKeyUp    0
#define ArrowKeyDown  1
#define ArrowKeyRight 2
#define ArrowKeyLeft  3

std::array<bool, UCHAR_MAX> keyState{}, keyStateBackup{};
constexpr float ANGLE_DELTA = 4;


// Camera
glm::vec3 cameraPosition(0, 1.5f, 0);
glm::vec3 cameraRotation(0, 0, 0);
float zNear = 0.1f, zFar = 100.f;

// Random Misc temporary testing things
int axisIndex;
std::array<LineSegment, 12> axes;

enum GeometryThing : unsigned short
{
	PlusX  = 1 << 1,
	MinusX = 1 << 2,
	PlusZ  = 1 << 3,
	MinusZ = 1 << 4,
	PlusY  = 1 << 5,
	MinusY = 1 << 6,
	WallX  = PlusX | MinusX,
	WallZ  = PlusZ | MinusZ,
	HallwayZ = PlusX | MinusX | PlusY,
	HallwayX = PlusZ | MinusZ | PlusY,
	All = 0xFF,
};

std::vector<Model> GetPlaneSegment(const glm::vec3& base, GeometryThing flags)
{
	std::vector<Model> results;
	if (flags & PlusX)  results.push_back({ base + glm::vec3(-1, 1,  0), glm::vec3(  0, 0, -90.f) });
	if (flags & MinusX) results.push_back({ base + glm::vec3( 1, 1,  0), glm::vec3(  0, 0,  90.f) });
	if (flags & PlusZ)  results.push_back({ base + glm::vec3( 0, 1, -1), glm::vec3( 90, 0,     0) });
	if (flags & MinusZ) results.push_back({ base + glm::vec3( 0, 1,  1), glm::vec3(-90, 0,     0) });
	if (flags & PlusY)  results.push_back({ base });
	if (flags & MinusY) results.push_back({ base + glm::vec3( 0, 2,  0), glm::vec3(180, 0,     0) });
	return results;
}

std::vector<Model> GetHallway(const glm::vec3& base, bool openZ = true)
{
	return GetPlaneSegment(base, (openZ) ? HallwayZ : HallwayX);
}

// TODO: Line Shader with width, all the math being on gpu (given the endpoints and the width then do the orthogonal to the screen kinda thing)
// TODO: Move cube stuff into a shader or something I don't know

OBB loom;


Shader fontShader;
VAO fontVAO;
Buffer<ArrayBuffer> boring;
ASCIIFont fonter;

bool flopper = false;
void display()
{
	auto fumod = fonter.Render("lets goo");
	fumod.GetColor().SetFilters(LinearLinear, MagLinear);

	depthed.Bind();
	glViewport(0, 0, 1000, 1000);
	glClearColor(0, 0, 0, 1);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// Camera matrix
	glm::vec3 angles2 = glm::radians(cameraRotation);

	// Adding pi/2 is necessary because the default camera is facing -z
	glm::mat4 view = glm::translate(glm::eulerAngleXYZ(angles2.x, angles2.y + glm::half_pi<float>(), angles2.z), -cameraPosition);
	cameraUniformBuffer.BufferSubData(view, 0);

	instancing.SetActiveShader();
	instancing.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	instancing.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	instancing.SetVec3("viewPos", cameraPosition);
	instancing.SetTextureUnit("textureIn", wallTexture, 0);
	instancing.SetTextureUnit("ditherMap", ditherTexture, 1);

	instanceVAO.BindArrayBuffer(instanceBuffer, 1);
	glBindVertexBuffer(0, texturedPlane.GetBuffer(), 0, sizeof(TextureVertex));
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei) planes.size());

	/* STICK FIGURE GUY */
	uniform.SetActiveShader();
	plainVAO.BindArrayBuffer(stickBuffer);

	glm::vec3 colors = glm::vec3(1, 0, 0);
	Model m22(glm::vec3(10, 0, 0));
	uniform.SetMat4("Model", m22.GetModelMatrix());
	uniform.SetVec3("color", colors);
	uniform.DrawIndexed<LineStrip>(stickIndicies);

	// Debugging boxes
	if (debugFlags[TIGHT_BOXES] || debugFlags[WIDE_BOXES])
	{
		glm::vec3 blue(0, 0, 1);
		plainVAO.BindArrayBuffer(plainCube);

		OBB goober(AABB(glm::vec3(0), glm::vec3(1)));
		goober.Translate(glm::vec3(2, 0.1, 0));
		goober.Rotate(glm::radians(glm::vec3(frameCounter * -2.f, frameCounter * 4.f, frameCounter)));
		uniform.SetMat4("Model", goober.GetModelMatrix());
		uniform.SetVec3("color", blue);

		float wid = 10;
		if (debugFlags[TIGHT_BOXES]) uniform.DrawIndexed<Lines>(cubeOutlineIndex);
		uniform.SetMat4("Model", goober.GetAABB().GetModel().GetModelMatrix());
		uniform.SetVec3("color", glm::vec3(0.5f, 0.5f, 0.5f));

		if (debugFlags[WIDE_BOXES]) uniform.DrawIndexed<Lines>(cubeOutlineIndex);
		for (const auto& box: boxes)
		{
			//glLineWidth((box.color) ? wid * 1.5f : wid);
			//glPointSize((box.color) ? wid * 1.5f : wid);
			if (debugFlags[TIGHT_BOXES])
			{
				uniform.SetMat4("Model", box.box.GetModelMatrix());
				uniform.SetVec3("color", (box.color) ? blue : colors);
				uniform.DrawIndexed<Lines>(cubeOutlineIndex);
				uniform.DrawElements<Points>(8);
			}
			if (debugFlags[WIDE_BOXES])
			{
				uniform.SetVec3("color", (box.color) ? colors : blue);
				uniform.SetMat4("Model", box.box.GetAABB().GetModel().GetModelMatrix());
				uniform.DrawIndexed<Lines>(cubeOutlineIndex);
				uniform.DrawElements<Points>(8);
			}
		}
	}

	// Cubert
	plainVAO.BindArrayBuffer(plainCube);
	uniform.SetMat4("Model", dumbBox.GetModelMatrix());
	
	uniform.SetVec3("color", glm::vec3(1, 1, 1));
	uniform.SetMat4("Model", loom.GetModelMatrix());
	//uniform.DrawIndexedMemory<Triangle>(cubeIndicies);

	// Albert
	/*
	texturedVAO.BindArrayBuffer(albertBuffer);
	dither.SetActiveShader();
	dither.SetTextureUnit("ditherMap", wallTexture, 1);
	dither.SetTextureUnit("textureIn", texture, 0);
	dither.SetMat4("Model", smartBox.GetModelMatrix());
	dither.SetVec3("color", (!smartBoxColor) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0));
	*/
	uniform.SetActiveShader();
	uniform.SetMat4("Model", smartBox.GetModelMatrix());
	uniform.DrawIndexed<Lines>(cubeOutlineIndex);

	// Drawing of the rays
	glDisable(GL_DEPTH_TEST);
	plainVAO.BindArrayBuffer(rayBuffer);
	Model bland;
	uniform.SetMat4("Model", bland.GetModelMatrix());
	glLineWidth(15.f);
	uniform.DrawElements<Lines>(rayBuffer);
	glLineWidth(1.f);
	glEnable(GL_DEPTH_TEST);

	// Sphere drawing
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	flatLighting.SetActiveShader();

	normalVAO.BindArrayBuffer(sphereBuffer);

	Model sphereModel(glm::vec3(6.5f, 5.5f, 0.f));
	sphereModel.translation += glm::vec3(0, 1, 0) * (float) glm::sin(glm::radians(frameCounter * 0.5f)) * 0.25f;
	sphereModel.scale = glm::vec3(1.5f);
	//sphereModel.rotation += glm::vec3(0.5f, 0.25, 0.125) * (float) frameCounter;
	sphereModel.rotation += glm::vec3(0, 0.25, 0) * (float) frameCounter;
	/*
	flatLighting.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	flatLighting.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	flatLighting.SetVec3("viewPos", cameraPosition);
	flatLighting.SetVec3("shapeColor", glm::vec3(1.f, .75f, 0.f));
	flatLighting.SetMat4("modelMat", sphereModel.GetModelMatrix());
	flatLighting.SetMat4("normMat", sphereModel.GetNormalMatrix());
	flatLighting.SetTextureUnit("hatching", hatching, 0);
	*/

	// Doing this while letting the normal be the color will create a cool effect
	
	sphereMesh.SetActiveShader();
	meshVAO.BindArrayBuffer(sphereBuffer);
	sphereMesh.SetMat4("modelMat", sphereModel.GetModelMatrix());
	sphereMesh.SetMat4("normalMat", sphereModel.GetNormalMatrix());
	sphereMesh.SetTextureUnit("textureIn", texture, 0);
	//mapper.BindTexture(0);
	//sphereMesh.SetTextureUnit("textureIn", 0);
	//sphereMesh.DrawIndexed<Triangle>(sphereIndicies);

	flatLighting.SetActiveShader();
	meshVAO.BindArrayBuffer(capsuleBuffer);
	flatLighting.SetVec3("lightColor", glm::vec3(1.f, 0.f, 0.f));
	flatLighting.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	flatLighting.SetVec3("viewPos", cameraPosition);
	flatLighting.SetMat4("modelMat", loom.GetNormalMatrix());
	flatLighting.SetMat4("normalMat", loom.GetNormalMatrix());
	//flatLighting.SetVec3("shapeColor", glm::vec3(0.8f, 0.34f, 0.6f));
	flatLighting.SetVec3("shapeColor", glm::vec3(0.f, 0.f, 0.8f));
	flatLighting.DrawIndexed<Triangle>(capsuleIndex);
	// Calling with triangle_strip is fucky
	/*
	flatLighting.DrawIndexed(Triangle, sphereIndicies);
	sphereModel.translation = moveSphere;
	flatLighting.SetMat4("modelMat", sphereModel.GetModelMatrix());
	flatLighting.SetMat4("normMat", sphereModel.GetNormalMatrix());
	flatLighting.DrawIndexed(Triangle, sphereIndicies);
	*/

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	uiRect.SetActiveShader();
	uiRect.SetVec4("color", glm::vec4(0, 0.5, 0.75, 0.25));
	uiRect.SetVec2("screenSize", glm::vec2(1000, 1000));
	uiRect.SetVec4("rectangle", glm::vec4(0, 0, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);
	
	uiRect.SetVec4("rectangle", glm::vec4(800, 0, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);
	
	uiRect.SetVec4("rectangle", glm::vec4(0, 900, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);
	
	uiRect.SetVec4("rectangle", glm::vec4(800, 900, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);

	uiRectTexture.SetActiveShader();
	Texture2D& ref = fumod.GetColor();
	uiRectTexture.SetVec4("rectangle", glm::vec4(200.f, 200.f, ref.GetWidth(), ref.GetHeight()));
	uiRectTexture.SetTextureUnit("image", ref, 0);
	uiRectTexture.DrawElements(TriangleStrip, 4);

	fontShader.SetActiveShader();
	fontVAO.BindArrayBuffer(boring);
	fontShader.SetTextureUnit("fontTexture", fonter.GetTexture(), 0);
	// TODO: Set object amount in buffer function
	fontShader.DrawElements<Triangle>(boring);
	glDisable(GL_BLEND);

	// Framebuffer stuff
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	scratchSpace.Bind();
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	// TODO: Uniformly lit shader with a "sky" light type of thing to provide better things idk
	
	/*
	frameShader.SetActiveShader();
	frameShader.SetTextureUnit("normal", depthed.GetColorBuffer<1>(), 0);
	frameShader.SetTextureUnit("depth", depthed.GetDepth(), 1);
	frameShader.SetFloat("zNear", zNear);
	frameShader.SetFloat("zFar", zFar);
	frameShader.SetInt("zoop", 0);
	frameShader.DrawElements<TriangleStrip>(4);
	*/
	BindDefaultFrameBuffer();
	glClearColor(1, 0.5, 0.25, 1);
	glClear(GL_DEPTH_BUFFER_BIT);

	expand.SetActiveShader();
	expand.SetTextureUnit("screen", depthed.GetColorBuffer<0>(), 0);
	expand.SetTextureUnit("edges", depthed.GetColorBuffer<1>(), 1);
	expand.SetTextureUnit("depths", depthed.GetDepth(), 2);
	expand.SetInt("depth", 5);
	frameShader.DrawElements<TriangleStrip>(4);

	// TODO: This is reversed <- wtf are you saying
	glLineWidth(1.f);
	widget.SetActiveShader();
	widget.DrawElements<Lines>(6);

	glFlush();
	glutSwapBuffers();
}

static const glm::vec3 GravityAxis{ 0.f, -1.f, 0.f };
static const glm::vec3 GravityUp{ 0.f, 1.f, 0.f };

struct
{
	glm::vec3 acceleration{ 0.f };
	glm::vec3 velocity{ 0.f };
	glm::vec3 axisOfGaming{ 0.f };
} smartBoxPhysics;

bool smartBoxCollide()
{
	bool val = false;
	smartBoxPhysics.axisOfGaming = glm::vec3{ 0.f };
	float dotValue = INFINITY;
	auto boxers = boxes.Search(smartBox.GetAABB());
	int collides = 0;
	//std::cout << "\r" << boxers.size();
	//Before(smartBoxPhysics.axisOfGaming);
	for (auto& letsgo : boxers)
	{
		//smartBox.OverlapAndSlide(letsgo->box);
		SlidingCollision c;
		if (smartBox.Overlap(letsgo->box, c))
		{
			float minDot = INFINITY, maxDot = -INFINITY;
			int minDotI = 0, maxDotI = 0;
			float minSign = 0, maxSign = 0;
			glm::vec3 upper = c.axis;
			for (std::size_t i = 0; i < 3; i++)
			{
				float sign = glm::sign(glm::dot(smartBox[i], upper));
				float local = glm::abs(glm::dot(smartBox[i], upper));
				if (local < minDot)
				{
					minDot = local;
					minDotI = i;
					minSign = sign;
				}
				if (local > maxDot)
				{
					maxDot = local;
					maxDotI = i;
					maxSign = sign;
				}
			}
			//if (glm::abs(maxDotI - 1) < EPSILON)
			{
				if (glm::abs(minSign) < EPSILON)
				{
					minSign = 1.f;
				}
				//std::cout << "Dots: " << minDotI << ", " << maxDotI << std::endl;
				//std::cout << "Signs" << minSign << ", " << maxSign << std::endl;
				glm::mat3 goobers{smartBox[0], smartBox[1], smartBox[2]};
				// Leasted aligned keeps its index
				// Middle is replaced with least cross intersection
				// Most is replaced with the negative of new middle cross least
				glm::vec3 least = smartBox[minDotI];                           // goes in smartbox[minDotI]
				glm::vec3 newMost = upper;                              // goes in smartbox[maxDotI]
				glm::vec3 newest = glm::normalize(glm::cross(least, newMost)); // goes in the remaining one(smartbox[3 - minDotI - maxDotI])

				//std::cout << least << "," << newMost << std::endl;
				int leastD = minDotI;
				int mostD = maxDotI;
				int newD = 3 - leastD - mostD;
				/*
				0, 1 -> no adjustment needed
				0, 2 -> multiplied by -1
				1, 0 -> multiplied by -1
				1, 2 -> no adjustment
				2, 0 -> no adjustment
				2, 1 -> multiplied by -1
				*/
				//if ((leastD == 0 && mostD == 2) || (leastD == 1 && mostD == 2) || (leastD == 2 && mostD == 1))
					//newest *= -1;
				least *= glm::sign(glm::dot(least, goobers[minDotI]));
				newMost *= glm::sign(glm::dot(newMost, goobers[maxDotI]));
				newest *= glm::sign(glm::dot(newest, goobers[newD]));
				//std::cout << glm::dot(newest, goobers[maxDotI]) << std::endl;
				// Correct for the wrong cross product ordering
				//if (leastD > mostD)
					//newest *= -1;

				// WIP
				//newest *= minSign;
				//newMost *= maxSign;

				glm::mat3 lame{};
				lame[leastD] = least;
				lame[mostD] = newMost;
				lame[newD] = newest;
				glm::mat4 newerst(lame);
				newerst[3].w = 1;
				glm::quat older = smartBox.GetNormalMatrix();
				glm::quat newer = newerst;
				older = glm::normalize(older);
				newer = glm::normalize(newer);
				float maxDelta = glm::acos(glm::abs(glm::dot(older, newer)));
				float clamped = std::clamp(c.depth / maxDelta, 0.f, 1.f);
				/*
				std::cout << maxDelta << "," << glm::degrees(maxDelta) <<  std::endl;
				std::cout << glm::acos(glm::dot(older, newer)) << std::endl;
				std::cout << older << std::endl;
				std::cout << newer << std::endl;
				*/
				if (glm::abs(glm::dot(older, newer) - 1) > EPSILON)
				{
					smartBox.ReOrient(glm::toMat4(glm::normalize(glm::lerp(older, newer, maxDelta / 2.f))));
				}
				std::array<glm::vec3, 12> rays{};
				rays.fill(glm::vec3(0));
				rays[1] = lame[0];
				rays[3] = lame[1];
				rays[5] = lame[2];

				glm::vec3 miniOff(0, 1, 0);
				rays[6] = miniOff;
				rays[7] = miniOff + goobers[0];
				rays[8] = miniOff;
				rays[9] = miniOff + goobers[1];
				rays[10] = miniOff;
				rays[11] = miniOff + goobers[2];
				rayBuffer.BufferData(rays, StaticDraw);
			}
			/*
			collides++;
			float orientation = glm::dot(GravityAxis, c.normal);
			if (glm::abs(orientation) > EPSILON && orientation < dotValue)
			{
				//std::cout << "OLD: " << smartBoxPhysics.axisOfGaming << "\tNew: " << c.normal << std::endl;
				dotValue = orientation;
				smartBoxPhysics.axisOfGaming = c.normal;
			}
			*/
			//Before(smartBox.Center());
			smartBox.ApplyCollision(c);
			//After(smartBox.Center());
			float oldLength = glm::length(smartBoxPhysics.velocity);
			//smartBoxPhysics.velocity += c.normal * glm::abs(glm::dot(smartBoxPhysics.velocity, c.normal));
			//if (glm::length(smartBoxPhysics.velocity) > EPSILON)
				//smartBoxPhysics.velocity = glm::normalize(smartBoxPhysics.velocity) * oldLength;
			val = true;
			//gamers.push_back({ &(letsgo->box), c });
		}
	}
	//After(smartBoxPhysics.axisOfGaming);
	//std::cout << ":" << collides;
	return val;
}

glm::quat oldMan{};
glm::quat newMan{};

// TODO: Mech suit has an interior for the pilot that articulates seperately from the main body, within the outer limits of the frame
// Like it's a bit pliable
void idle()
{
	static auto lastTimers = std::chrono::high_resolution_clock::now();
	frameCounter++;
	const auto now = std::chrono::high_resolution_clock::now();
	const auto delta = now - lastTimers;
	static std::deque<float> frames;

	OBB goober2(AABB(glm::vec3(0), glm::vec3(1)));
	goober2.Translate(glm::vec3(2, 0.1, 0));	
	goober2.Rotate(glm::radians(glm::vec3(0, frameCounter * 4.f, 0)));
	glm::mat4 tester = goober2.GetNormalMatrix();

	//std::cout << "\r" << goober2.Forward() << "\t" << goober2.Cross() << "\t" << goober2.Up();
	//std::cout << "\r" << "AABB Axis: " << goober2.Forward() << "\t Euler Axis" << tester * glm::vec4(1, 0, 0, 0) << std::endl;
	//std::cout << "\r" << "AABB Axis: " << goober2.Forward() << "\t Euler Axis" << glm::transpose(tester)[0];

	Plane foobar(glm::vec3(1, 0, 0), glm::vec3(4, 0, 0)); // Facing away from origin
	//foobar.ToggleTwoSided();
	//if (!smartBox.IntersectionWithResponse(foobar))
		//sacounter++;
		//std::cout << counter << std::endl;
	//smartBox.RotateAbout(glm::vec3(0.05f, 0.07f, -0.09f), glm::vec3(0, -5, 0));
	//smartBox.RotateAbout(glm::vec3(0, 0, 0.05f), glm::vec3(0, -2, 0));
	//smartBox.RotateAbout(glm::vec3(0.05f, 0, 0), glm::vec3(0, -5, 0));

	//float speed = 3 * ((float) elapsed) / 1000.f;
	const float timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(delta).count();
	
	// TODO: Rolling buffer size thingy setting
	frames.push_back(1.f / timeDelta);
	if (frames.size() > 100)
	{
		frames.pop_front();
	}
	float averageFps = 0.f;
	for (auto& i : frames)
	{
		averageFps += i / frames.size();
	}
	fonter.RenderToScreen(boring, 0, 0, std::format("FPS:{:7.2f}\n{}", averageFps, (flopper) ? "true" : "false"));
	// End of Rolling buffer

	float speed = 4 * timeDelta;
	float turnSpeed = 100 * timeDelta;

	glm::vec3 forward = glm::eulerAngleY(glm::radians(-cameraRotation.y)) * glm::vec4(1, 0, 0, 0);
	glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
	forward = speed * glm::normalize(forward);
	right = speed * glm::normalize(right);
	glm::vec3 previous = cameraPosition;
	if (keyState['z'])
	{
		dumbBox.Translate(dumbBox.Forward() * speed);
	}
	if (keyState['x'])
	{
		dumbBox.Translate(dumbBox.Forward() * -speed);
	}
	if (keyState[ArrowKeyRight]) smartBox.Rotate(glm::vec3(0, -1.f, 0) * turnSpeed);
	if (keyState[ArrowKeyLeft])  smartBox.Rotate(glm::vec3(0, 1.f, 0) * turnSpeed);
	if (keyState['p'] || keyState['P'])
		std::cout << previous << std::endl;
	if (keyState['w'] || keyState['W'])
		cameraPosition += forward;
	if (keyState['s'] || keyState['S'])
		cameraPosition -= forward;
	if (keyState['d'] || keyState['D'])
		cameraPosition += right;
	if (keyState['a'] || keyState['A'])
		cameraPosition -= right;
	if (keyState['z']) cameraPosition += GravityUp * speed;
	if (keyState['x']) cameraPosition -= GravityUp * speed;
	if (cameraPosition != previous)
	{
		AABB playerBounds(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
		OBB goober(playerBounds);
		playerBounds.Center(cameraPosition);

		OBB playerOb(playerBounds);
		playerOb.Rotate(glm::eulerAngleY(glm::radians(-cameraRotation.y)));
		//playerOb.Rotate(glm::radians(glm::vec3(0, 45, 0)));

		goober.Translate(glm::vec3(2, 0, 0));
		goober.Rotate(glm::radians(glm::vec3(0, frameCounter * 4.f, 0)));
		for (auto& wall : boxes)
		{
			if (wall.box.Overlap(playerOb))
			{
				playerOb.OverlapAndSlide(wall.box);
				//offset = previous;
				//break;
			}
		}
		if (goober.Overlap(playerOb))
		{
			//offset = previous;
		}
		cameraPosition = playerOb.Center();
		//Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f))
	}

	// Physics attempt
	smartBoxPhysics.acceleration = glm::vec3(0.f);
	const float BoxAcceleration = 0.06f;// 0.06f;
	const float BoxMass = 1;
	const float staticFrictionCoeff = 1.0f;
	const float slidingFrictionCoeff = 0.57f;
	const float GravityRate = 1.f;
	const float BoxGravityMagnitude = BoxMass * GravityRate;
	const glm::vec3 boxGravity = GravityAxis * BoxGravityMagnitude;
	const float cose = glm::abs(glm::dot(smartBoxPhysics.axisOfGaming, GravityUp));
	const float sine = glm::sqrt(1.f - cose * cose);
	const glm::vec3 oldNormal = smartBoxPhysics.axisOfGaming;

	glm::vec3 boxForces{ 0.f };

	if (keyState[ArrowKeyUp])   boxForces += smartBox.Forward() * BoxAcceleration;
	if (keyState[ArrowKeyUp])   smartBox.Translate(smartBox.Forward() * speed);
	if (keyState[ArrowKeyDown]) boxForces -= smartBox.Forward() * BoxAcceleration;
	if (keyState[ArrowKeyDown]) smartBox.Translate(-smartBox.Forward() * speed);
	// Box is colliding with *something* pointing up
	//std::cout << smartBoxPhysics.axisOfGaming << std::endl;
	glm::vec3 lineThing = GravityAxis * smartBox.ProjectionLength(GravityAxis) * 1.1f; // Extend to account for slopes a bit
	AABB lineBox(smartBox.Center(), smartBox.Center() + lineThing);
	bool gab = false;

	if (frameCounter % 2000 == 0)
	{
		std::swap(oldMan, newMan);
	}
	unsigned int modded = frameCounter % 2000;
	//smartBox.ReOrient(glm::toMat4(glm::lerp(oldMan, newMan, modded / 2000.f)));

	for (auto& hit : boxes.Search(lineBox))
	{
		auto& hitter = hit->box;
		RayCollision rayd{};
		//std::cout << std::boolalpha << hitter.Intersect(smartBox.Center(), smartBox.Center() + lineThing) << std::endl;
		if (hitter.Intersect(smartBox.Center(), lineThing, rayd))
		{
			smartBoxPhysics.axisOfGaming = rayd.normal;
			gab = true;
			hit->color = true;
			//std::cout << "GOT ONE" << frameCounter << std::endl;
			break;
		}
	}

	if (gab)//(smartBoxPhysics.axisOfGaming != glm::vec3(0.f))
	{
		//                      Direction                      Magnitude
		glm::vec3 normalForce = smartBoxPhysics.axisOfGaming * BoxGravityMagnitude * cose;

		//std::cout << glm::tan(glm::acos(cose)) << std::endl;

		// If 
		if (staticFrictionCoeff < glm::tan(glm::acos(cose)))
		{
			boxForces += boxGravity - normalForce;
		}
		if (glm::length(smartBoxPhysics.velocity) > EPSILON)
		{
			//std::cout << "\rMOVING";
			//boxForces += glm::normalize(-smartBoxPhysics.velocity) * BoxGravityMagnitude * cose * slidingFrictionCoeff;
		}
		else
		{
			//std::cout << "\rSTOPPED";q
		}


		// This is the "up the slope" vector, sine is removed due to it being irrelevent, but it would be properly scaled there
		glm::vec3 friction{ 0.f };
		/*
		// TODO: Make sure that if it's already sliding to ignore this
		if (smartBoxPhysics.axisOfGaming != GravityUp)// && glm::length(smartBoxPhysics.velocity) < EPSILON)
		{
			// WORKING VERSION
			//friction = glm::normalize(GravityUp - smartBoxPhysics.axisOfGaming * cose) * glm::length(normalForce) * staticFrictionCoeff;
			friction = glm::normalize(GravityUp - smartBoxPhysics.axisOfGaming) * glm::min(cose * staticFrictionCoeff, sine) * BoxGravityMagnitude;
		}
		*/
		if (glm::length(smartBoxPhysics.velocity) > EPSILON)
		{
			//friction += glm::normalize(-smartBoxPhysics.velocity) * glm::length(normalForce) * slidingFrictionCoeff;
		}
		//boxForces += normalForce + friction;
	}
	else
	{
		//std::cout << "Denied " << frameCounter << std::endl;
		boxForces += boxGravity;
	}
	// A = F / M
	//std::cout << boxForces << std::endl;
	//std::cout << boxForces << std::endl;
	smartBoxPhysics.acceleration = boxForces / BoxMass * timeDelta;

	smartBoxPhysics.velocity += smartBoxPhysics.acceleration;
	if (glm::length(smartBoxPhysics.velocity) > 2.f)
		smartBoxPhysics.velocity = glm::normalize(smartBoxPhysics.velocity) * 2.f;
	//smartBox.Translate(smartBoxPhysics.velocity);
	smartBoxPhysics.velocity *= 0.99f;

	smartBoxColor = smartBoxCollide();
	if (staticFrictionCoeff >= glm::tan(glm::acos(cose)))
	{
		//smartBoxPhysics.axisOfGaming = oldNormal;
	}
	axes = smartBox.GetLineSegments();

	//smartBox.OverlapCompleteResponse(dumbBox);

	Sphere awwYeah(0.5f, moveSphere);
	for (auto& letsgo : boxes.Search(AABB(awwYeah.center - glm::vec3(awwYeah.radius), awwYeah.center + glm::vec3(awwYeah.radius))))
	{
		Collision c;
		if (letsgo->box.Overlap(awwYeah, c))
		{
			awwYeah.center -= c.normal * c.depth;
		}
	}
	moveSphere = awwYeah.center;
	
	// TODO: Key state is ten kinds of messed up
	std::copy(std::begin(keyState), std::end(keyState), std::begin(keyStateBackup));
	std::swap(keyState, keyStateBackup);

	lastTimers = now;
	glutPostRedisplay();
}

void smartReset()
{
	smartBox.ReCenter(glm::vec3(2, 1.f, 0));
	smartBox.ReOrient(glm::vec3(0, 135, 0));
}

void keyboard(unsigned char key, int x, int y)
{
	// TODO: Whole key thing needs to be re-written
	keyState[key] = true;
	if (key == 'm' || key == 'M') cameraPosition.y += 3;
	if (key == '[') lineWidth -= 1;
	if (key == ']') lineWidth += 1;
	if (key == 'n' || key == 'N') cameraPosition.y -= 3;
	if (key == 'q' || key == 'Q') glutLeaveMainLoop();
	if (key == 't' || key == 'T') kernel = 1 - kernel;
	if (key == 'b') flopper = !flopper;
	if (key == 'h' || key == 'H')
	{
		smartBox.ReOrient(glm::vec3(0, 89, 0));
		smartBox.ReCenter(glm::vec3(-0.625, 1.f, 0));
	}
	if (key >= '1' && key <= '9')
	{
		std::size_t value = (std::size_t) key - '0';
		debugFlags[value] = !debugFlags[value];
	}
	if (key == 'r' || key == 'R')
	{
		axisIndex = (axisIndex + 1) % 12;
		if (axes.size() > axisIndex)
		{
			//std::array<glm::vec3, 2> verts = { axes[axisIndex].A, axes[axisIndex].B };
			//rayBuffer.BufferSubData(verts);
		}

		glm::vec3 angles2 = glm::radians(cameraRotation);

		glm::vec3 gamer = glm::normalize(glm::eulerAngleXYZ(-angles2.z, -angles2.y, -angles2.x) * glm::vec4(1, 0, 0, 0));
		std::array<glm::vec3, 8> verts = { cameraPosition, cameraPosition + gamer * 100.f , cameraPosition, cameraPosition + gamer * 100.f };

		bool set = false;


		RayCollision nears, fars;
		//smartBox.Intersect(cameraPosition, gamer, nears, fars);
		/*
		auto foosball = smartBox.ClosestFacePoints(cameraPosition);
		std::array<glm::vec3, 12> localpoints;
		for (std::size_t i = 0; i < localpoints.size() && i < foosball.size(); i++)
		{
			localpoints[i] = foosball[i];
		}
		rayBuffer.BufferSubData(localpoints);
		*/
		//for (std::size_t i = 0; i < boxes.size(); i++)

		for (auto& box: boxes)
		{
			//boxColor[i] = boxes[i].Intersect(offset, gamer * 100.f, nears, fars);
			box.color = box.box.Intersect(cameraPosition, gamer * 100.f, nears, fars);
			if (box.color && !set)
			{
				set = true;
				glm::vec3 point = cameraPosition + gamer * nears.distance * 100.f;
				for (std::size_t j = 0; j < 3; j++)
				{
					verts[2 + 2 * j] = point;
					glm::vec3 cur = glm::normalize(box.box[j]);
					verts[2 + 2 * j + 1] = point + SlideAlongPlane(cur, gamer) * 100.f;//point + glm::normalize(gamer - glm::dot(gamer, cur) * cur) * 100.f;
				}
				break;
			}
		}
		//std::cout << glm::vec3(-1, 1.2f, -2) << "->" << dumbBox.Center() << std::endl;
		//bool flopped = dumbBox.Intersect(glm::vec3(-1, 1.2f, -2), glm::normalize(glm::vec3(2, 0.f, 0)), nears, fars);
		// dumbBox.Intersect(glm::vec3(-1, 1.2f, -2), (glm::vec3(2, 0.f, 0)), nears, fars);
		//if (flopped || !flopped)
		//{
		//	std::cout << std::boolalpha << flopped << std::endl;
		//	//glm::vec3 point = cameraPosition + gamer * nears.distance * 100.f;
		//	glm::vec3 point = glm::vec3(-1, 1.2f, -2) + (glm::vec3(2, 0.1f, 0) * 100.f);
		//	verts[0] = point;
		//	verts[1] = glm::vec3(-1, 1.2f, -2);
		//	/*
		//	for (std::size_t j = 0; j < 3; j++)
		//	{
		//		verts[2 + 2 * j] = point;
		//		glm::vec3 cur = glm::normalize(dumbBox[j]);
		//		verts[2 + 2 * j + 1] = point + SlideAlongPlane(cur, glm::vec3(2, 0, 0)) * 100.f;
		//	}*/
		//}
		//rayBuffer.BufferSubData(verts);
	}
}

void keyboardOff(unsigned char key, int x, int y)
{
	keyState[key] = false;
}

static const float Fov = 70.f;
static bool rightMouseHeld = false;
static int mousePreviousX = 0, mousePreviousY = 0;

void mouseButtonAction(int button, int state, int x, int y)
{
	if (button == GLUT_RIGHT_BUTTON)
	{
		rightMouseHeld = (state == GLUT_DOWN);
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		glm::vec3 localCopy = cameraRotation;
		localCopy.x += (float)(y - glutGet(GLUT_WINDOW_WIDTH) / 2.f) / glutGet(GLUT_WINDOW_WIDTH) * Fov;
		localCopy.y += (float)(x - glutGet(GLUT_WINDOW_HEIGHT) / 2.f) / glutGet(GLUT_WINDOW_HEIGHT) * Fov;
		glm::vec3 radians = glm::radians(localCopy);

		float rayLength = 50.f;

		auto lookyMat = glm::eulerAngleXYZ(-radians.z, -radians.y, -radians.x);

		Ray liota(cameraPosition, glm::normalize(glm::vec3(lookyMat * glm::vec4(1, 0, 0, 0))));
		auto itemed = boxes.RayCast(liota);
		RayCollision rayd{};
		Dummy* point = nullptr;
		for (auto& item : itemed)
		{
			if (item->box.Intersect(liota.initial, liota.delta, rayd) && rayd.depth > 0.f && rayd.depth < rayLength)
			{
				rayLength = rayd.depth;
				point = &(*item);
			}
		}
		// Point now has the pointer to the closest element
		Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.1f, rayLength - 0.5f - 0.2f, 30, 30);
		loom.ReOrient(lookyMat);
		loom.ReScale(glm::vec3((rayLength - 0.5f) / 2.f, 0.1f, 0.1f));

		loom.ReCenter(cameraPosition);
		loom.Translate(loom.Forward() * (0.3f + rayLength / 2.f));
	}
}

void mouseMovementFunction(int x, int y)
{
	if (rightMouseHeld)
	{
		float xDif = (float)(x - mousePreviousX);
		float yDif = (float)(y - mousePreviousY);
		if (abs(xDif) > 20)
			xDif = 0;
		if (abs(yDif) > 20)
			yDif = 0;
		float yDelta = 50 * (xDif * ANGLE_DELTA) / glutGet(GLUT_WINDOW_WIDTH);
		float zDelta = 50 * (yDif * ANGLE_DELTA) / glutGet(GLUT_WINDOW_HEIGHT);

		cameraRotation.x += zDelta;
		cameraRotation.y += yDelta;
	}
	mousePreviousX = x;
	mousePreviousY = y;
}

void specialKeys(int key, [[maybe_unused]] int x, [[maybe_unused]] int y)
{
	// TODO: Investigate stuff relating to number pad keys
	switch (key)
	{
	case GLUT_KEY_UP: keyState[ArrowKeyUp] = true; break;
	case GLUT_KEY_DOWN: keyState[ArrowKeyDown] = true; break;
	case GLUT_KEY_RIGHT: keyState[ArrowKeyRight] = true; break;
	case GLUT_KEY_LEFT: keyState[ArrowKeyLeft] = true; break;
	case GLUT_KEY_F1: debugFlags[TIGHT_BOXES] = !debugFlags[TIGHT_BOXES]; break;
	case GLUT_KEY_F2: debugFlags[WIDE_BOXES] = !debugFlags[WIDE_BOXES]; break;
	default: break;
	}
}

void specialKeysUp(int key, [[maybe_unused]] int x, [[maybe_unused]] int y)
{
	switch (key)
	{
	case GLUT_KEY_UP: keyState[ArrowKeyUp] = false; break;
	case GLUT_KEY_DOWN: keyState[ArrowKeyDown] = false; break;
	case GLUT_KEY_RIGHT: keyState[ArrowKeyRight] = false; break;
	case GLUT_KEY_LEFT: keyState[ArrowKeyLeft] = false; break;
	default: break;
	}
}

void DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	// FROM https://gist.github.com/liam-middlebrook/c52b069e4be2d87a6d2f
	const char* _source;
	const char* _type;
	const char* _severity;

	switch (source) {
	case GL_DEBUG_SOURCE_API: _source = "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM: _source = "WINDOW SYSTEM"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: _source = "SHADER COMPILER"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY: _source = "THIRD PARTY"; break;
	case GL_DEBUG_SOURCE_APPLICATION: _source = "APPLICATION"; break;
	case GL_DEBUG_SOURCE_OTHER: _source = "UNKNOWN"; break;
	default: _source = "UNKNOWN"; break;
	}

	switch (type) {
	case GL_DEBUG_TYPE_ERROR: _type = "ERROR"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:_type = "DEPRECATED BEHAVIOR"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:_type = "UDEFINED BEHAVIOR"; break;
	case GL_DEBUG_TYPE_PORTABILITY:_type = "PORTABILITY"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:_type = "PERFORMANCE"; break;
	case GL_DEBUG_TYPE_OTHER:_type = "OTHER"; break;
	case GL_DEBUG_TYPE_MARKER:_type = "MARKER"; break;
	default:_type = "UNKNOWN"; break;
	}

	switch (severity) {
	case GL_DEBUG_SEVERITY_HIGH:_severity = "HIGH"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:_severity = "MEDIUM"; break;
	case GL_DEBUG_SEVERITY_LOW:_severity = "LOW"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION:_severity = "NOTIFICATION"; break;
	default:_severity = "UNKNOWN"; break;
	}

	printf("%d: %s of %s severity, raised from %s: %s\n",
		id, _type, _severity, _source, message);
}

int main(int argc, char** argv)
{
	int error = 0;
	debugFlags.fill(false);
	// Glut
	glutInit(&argc, argv);
	// TODO: Maybe transition to OpenGL ES3.1
	glutInitContextVersion(4, 6);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(1000, 1000);
	glutInitContextFlags(GLUT_DEBUG);
	glutCreateWindow("Wowie a window");
	glViewport(0, 0, 1000, 1000);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	//glDisable(GL_LINE_SMOOTH);
	//glDisable(GL_POLYGON_SMOOTH);

	//glDepthFunc(GL_LESS);
	glDepthFunc(GL_LEQUAL);
	glClearColor(0, 0, 0, 1);

	glFrontFace(GL_CCW);

	glutDisplayFunc(display);
	glutIdleFunc(idle);

	glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardOff);
	glutSpecialFunc(specialKeys);
	glutSpecialUpFunc(specialKeysUp);

	glutMouseFunc(mouseButtonAction);
	glutMotionFunc(mouseMovementFunction);
	glutPassiveMotionFunc(mouseMovementFunction);
	glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);
	glutPositionWindow(0, 0);


	glewExperimental = GL_TRUE;
	// Glew
	if ((error = glewInit()) != GLEW_OK)
	{
		printf("Error code %i from glewInit()", error);
		return -1;
	}
	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEBUG_OUTPUT);

	CheckError();
	glDebugMessageCallback(DebugCallback, nullptr);
	// Get rid of Line_width_deprecated messages
	GLuint toDisable = 7; 
	glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, GL_DONT_CARE, 1, &toDisable, GL_FALSE);


	// TODO: This noise stuff idk man
	//Shader::IncludeInShaderFilesystem("FooBarGamer.gsl", "uniformv.glsl");
	//Shader::IncludeInShaderFilesystem("noise2D.glsl", "noise2D.glsl");


	// SHADER SETUP
	Shader::SetBasePath("Shaders/");
	dither.CompileSimple("light_text_dither");
	expand.Compile("framebuffer", "expand");
	fontShader.CompileSimple("font");
	flatLighting.CompileSimple("lightflat");
	frameShader.CompileSimple("framebuffer");
	instancing.CompileSimple("instance");
	sphereMesh.CompileSimple("mesh");
	uiRect.CompileSimple("ui_rect");
	uiRectTexture.CompileSimple("ui_rect_texture");
	uniform.CompileSimple("uniform");
	widget.CompileSimple("widget");

	uniform.UniformBlockBinding("Camera", 0);
	dither.UniformBlockBinding("Camera", 0);
	flatLighting.UniformBlockBinding("Camera", 0);
	instancing.UniformBlockBinding("Camera", 0);
	sphereMesh.UniformBlockBinding("Camera", 0);

	uiRect.UniformBlockBinding("ScreenSpace", 1);
	uiRectTexture.UniformBlockBinding("ScreenSpace", 1);
	fontShader.UniformBlockBinding("ScreenSpace", 1);

	// VAO SETUP
	// TODO: maybe do away with this annoying generate(); fill() pattern, it's annoying
	fontVAO.Generate();
	fontVAO.FillArray<UIVertex>(fontShader);
	
	instanceVAO.Generate();
	instanceVAO.FillArray<TextureVertex>(instancing, 0);
	instanceVAO.FillArray<glm::mat4>(instancing, 1);
	instanceVAO.BufferBindingPointDivisor(0, 0);
	instanceVAO.BufferBindingPointDivisor(1, 1);

	meshVAO.Generate();
	meshVAO.FillArray<MeshVertex>(sphereMesh);

	normalVAO.Generate();
	normalVAO.FillArray<NormalVertex>(flatLighting);

	plainVAO.Generate();
	plainVAO.FillArray<Vertex>(uniform);

	texturedVAO.Generate();
	texturedVAO.FillArray<TextureVertex>(dither);

	// TEXTURE SETUP
	// TODO: texture loading base path thingy
	ditherTexture.Load(dither16, InternalRed, FormatRed, DataUnsignedByte);
	ditherTexture.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);
	hatching.Load("Textures/hatching.png");
	hatching.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);
	texture.Load("Textures/text.png");
	texture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);
	wallTexture.Load("Textures/flowed.png");
	wallTexture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);

	/*
	mapper.Generate({ "Textures/skybox/right.jpg", "Textures/skybox/left.jpg", "Textures/skybox/top.jpg",
		"Textures/skybox/bottom.jpg", "Textures/skybox/front.jpg", "Textures/skybox/back.jpg" });
	*/


	stickBuffer.Generate();
	stickBuffer.BufferData(stick, StaticDraw);

	std::array<TextureVertex, 4> verts{};
	for (int i = 0; i < 4; i++)
		verts[i].position = plane[i];
	verts[0].coordinates = glm::vec2(1, 1);
	verts[1].coordinates = glm::vec2(1, 0);
	verts[2].coordinates = glm::vec2(0, 1);
	verts[3].coordinates = glm::vec2(0, 0);
	texturedPlane.Generate();
	texturedPlane.BufferData(verts, StaticDraw);

	planeBO.Generate();
	planeBO.BufferData(plane, StaticDraw);

	plainCube.Generate();
	plainCube.BufferData(plainCubeVerts, StaticDraw);


	// RAY SETUP
	std::array<glm::vec3, 20> rays = {};
	rays.fill(glm::vec3(0.f));
	glm::vec3 a{ 2, 1, 1 };
	glm::vec3 fooey = glm::normalize(a);

	// Despite the dot product usually being the cosine bewtween the angles this is wrong due to the coordinate orientation hackery
	float aCose = glm::dot(fooey, glm::vec3(0, 1, 0));
	float aSine = glm::sqrt(1 - aCose * aCose);
	//std::cout << aSine << ":" << aCose << std::endl;
	rays[1] = fooey;
	rays[3] = GravityAxis;
	rays[5] = glm::normalize(glm::vec3(0, 1, 0) - fooey * aCose) * aSine;
	rays[9] = glm::vec3(-1, 2, 0);
	//std::cout << ":" << glm::length(fooey * aCose) << ":" << glm::length(rays[5]) << std::endl;
	rays.fill(glm::vec3(0.f));
	//std::cout << glm::dot(rays[5], fooey) << "\t" << EPSILON << "\t" << std::numeric_limits<float>::epsilon() << std::endl;
	/*
	gobs.fill(glm::vec3(0, -5, 0));
	gobs[0] += glm::vec3(1, 0, 0);
	gobs[1] += glm::vec3(-1, 0, 0);
	gobs[2] += glm::vec3(0, 0, 1);
	gobs[3] += glm::vec3(0, 0, -1);
	*/

	rayBuffer.Generate();
	rayBuffer.BufferData(rays, StaticDraw);


	// CREATING OF THE PLANES

	for (int i = -5; i <= 5; i++)
	{
		if (abs(i) <= 1)
			continue;
		CombineVector(planes, GetHallway(glm::vec3(0, 0, 2 * i), true));
		CombineVector(planes, GetHallway(glm::vec3(2 * i, 0, 0), false));
	}
	for (int i = 0; i < 9; i++)
	{
		CombineVector(planes, GetPlaneSegment(glm::vec3(2 * (i % 3 - 1), 0, 2 * (((int)i / 3) - 1)), PlusY));
	}
	planes.push_back(Model(glm::vec3( 2, 1.f, -2), glm::vec3(0,  45,  90.f), glm::vec3(1, 1, (float) sqrt(2))));
	planes.push_back(Model(glm::vec3( 2, 1.f,  2), glm::vec3(0, -45,  90.f), glm::vec3(1, 1, (float) sqrt(2))));
	planes.push_back(Model(glm::vec3(-2, 1.f,  2), glm::vec3(0,  45, -90.f), glm::vec3(1, 1, (float) sqrt(2))));
	planes.push_back(Model(glm::vec3(-2, 1.f, -2), glm::vec3(0, -45, -90.f), glm::vec3(1, 1, (float) sqrt(2))));

	planes.push_back(Model(glm::vec3(3.8f, .25f, 0), glm::vec3(0, 0.f, 15.0f), glm::vec3(1, 1, 1)));

	std::vector<glm::mat4> awfulTemp{};
	awfulTemp.reserve(planes.size());
	//planes.push_back(Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f)));
	for (const auto& ref : planes)
	{
		OBB project(ref);
		project.Scale(glm::vec3(1, .0625f, 1));
		boxes.Insert({project, false}, project.GetAABB());
		awfulTemp.push_back(ref.GetModelMatrix());
	}

	instanceBuffer.Generate();
	instanceBuffer.BufferData(awfulTemp, StaticDraw);

	// This sucks
	std::array<TextureVertex, 36> textVert{};
	for (std::size_t i = 0; i < 36; i++)
	{
		textVert[i].position = texturedCubeVerts[i];
		int j = i % 6;
		// j = 0/4 are unique, j = 1/2 are repeated as 3/5 respectively
		switch (j)
		{
		case 0: textVert[i].uvs = glm::vec2(0, 0); break;
		case 4: textVert[i].uvs = glm::vec2(1, 1); break;
		case 1: case 3: textVert[i].uvs = glm::vec2(0, 1); break;
		case 2: case 5: textVert[i].uvs = glm::vec2(1, 0); break;
		default: break;
		}
		// Need to rotate them
		if (i / 6 < 3)
		{
			// Ensure opposite sides have the same alignment
			textVert[i].uvs = textVert[i].uvs - glm::vec2(0.5f);
			textVert[i].uvs = glm::mat2(0, -1, 1, 0) * textVert[i].uvs;
			textVert[i].uvs = textVert[i].uvs + glm::vec2(0.5f);
		}
	}
	albertBuffer.Generate();
	albertBuffer.BufferData(textVert, StaticDraw);

	// FRAMEBUFFER SETUP
	// TODO: Renderbuffer for buffers that don't need to be directly read
	depthed.GetColorBuffer<0>().CreateEmpty(1000, 1000, InternalRGBA);
	depthed.GetColorBuffer<0>().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	depthed.GetColorBuffer<1>().CreateEmpty(1000, 1000, InternalRGBA);
	depthed.GetColorBuffer<1>().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	depthed.GetDepth().CreateEmpty(1000, 1000, InternalDepthFloat32);
	depthed.GetDepth().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	depthed.Assemble();

	scratchSpace.GetColorBuffer().CreateEmpty(1000, 1000, InternalRGBA);
	scratchSpace.GetColorBuffer().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);
	scratchSpace.Assemble();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	Sphere::GenerateMesh(sphereBuffer, sphereIndicies, 30, 30);
	Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.1f, 10.f, 30, 30);

	Font::SetFontDirectory("Fonts");

	// Awkward syntax :(
	ASCIIFont::LoadFont(fonter, "CommitMono-400-Regular.ttf", 50.f, 2, 2);

	stickIndicies.Generate();
	stickIndicies.BufferData(stickDex, StaticDraw);

	cubeOutlineIndex.Generate();
	cubeOutlineIndex.BufferData(cubeOutline, StaticDraw);

	smartBox.Scale(glm::vec3(0.5f));
	smartReset();
	smartBox.ReCenter(glm::vec3(1.2f, 0.6f, 0));
	smartBox.ReOrient(glm::vec3(0, 0, 0));
	dumbBox.ReCenter(glm::vec3(0, 1.f, -2));
	dumbBox.Scale(glm::vec3(1.f));
	dumbBox.Rotate(glm::vec3(0, -90, 0));

	loom.ReCenter(glm::vec3(0, 5, 0));
	//boxes.Insert({ dumbBox, false }, dumbBox.GetAABB());

	OBB checked(Model(glm::vec3(3.8f, .25f, 0), glm::vec3(0, 0.f, 15.0f), glm::vec3(1, 1, 1)));
	checked.Scale(glm::vec3(1, 0.0625f, 1));
	rays.fill(glm::vec3(0));
	float minDot = INFINITY, maxDot = -INFINITY;
	int minDotI = 0, maxDotI = 0;
	for (std::size_t i = 0; i < 3; i++)
	{
		float local = glm::abs(glm::dot(smartBox[i], checked.Up()));
		if (local < minDot)
		{
			minDot = local;
			minDotI = i;
		}
		if (local > maxDot)
		{
			maxDot = local;
			maxDotI = i;
		}
		//rays[6 + i * 2] = smartBox[i];
	}
	// Leasted aligned keeps its index
	// Middle is replaced with least cross intersection
	// Most is replaced with the negative of new middle cross least
	glm::vec3 least = smartBox[minDotI];                           // goes in smartbox[minDotI]
	glm::vec3 newMost = checked.Up();                              // goes in smartbox[maxDotI]
	glm::vec3 newest = glm::normalize(glm::cross(least, newMost)); // goes in the remaining one(smartbox[3 - minDotI - maxDotI])

	int leastD = minDotI;
	int mostD = maxDotI;
	int newD = 3 - leastD - mostD;

	//std::cout << "L:" << minDotI << "," << smartBox[minDotI] << std::endl << "M:" << maxDotI << ", " << smartBox[maxDotI] << std::endl;
	rays[1] = smartBox[minDotI];
	rays[5] = glm::normalize(glm::cross(smartBox[minDotI], checked.Up()));
	rays[3] = glm::normalize(glm::cross(rays[5], rays[1]));

	std::swap(rays[1], rays[5]);

	glm::mat3 lame{};
	//rays[1] *= -1;

	rays[2 * leastD + 1] = least;
	rays[2 * mostD + 1]  = newMost;
	rays[2 * newD + 1]   = newest;
	if (leastD > mostD)
	{
		rays[2 * newD + 1] *= -1;
	}
	lame[leastD] = least;
	lame[mostD] = newMost;
	lame[newD] = rays[2 * newD + 1];
	glm::mat4 newerst(lame);
	newerst[3].w = 1;
	oldMan = smartBox.GetModelMatrix();
	newMan = newerst;
	//smartBox.ReOrient(newerst);

	rayBuffer.BufferData(rays, StaticDraw);

	CheckError();

	screenSpaceBuffer.Generate(StaticRead, sizeof(glm::mat4));
	screenSpaceBuffer.SetBindingPoint(1);
	screenSpaceBuffer.BindUniform();
	screenSpaceBuffer.BufferSubData(glm::ortho<float>(0, 1000, 1000, 0));
	
	cameraUniformBuffer.Generate(DynamicDraw, 2 * sizeof(glm::mat4));
	cameraUniformBuffer.SetBindingPoint(0);
	cameraUniformBuffer.BindUniform();

	glm::mat4 projection = glm::perspective(glm::radians(Fov), 1.f, zNear, zFar);
	cameraUniformBuffer.BufferSubData(projection, sizeof(glm::mat4));
	CheckError();

	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glClearColor(0.f, 0.f, 0.f, 0.f);
	CheckError();
	glutMainLoop();

	return 0;
}