#include <algorithm>
#include <chrono>
#include <glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/hash.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <sys/utime.h>
#include <time.h>
#include <unordered_map>
#include "AABB.h"
#include "Buffer.h"
#include "Button.h"
#include "CubeMap.h"
#include "Font.h"
#include "Framebuffer.h"
#include "glmHelp.h"
#include "glUtil.h"
#include "Lines.h"
#include "log.h"
#include "Model.h"
#include "OrientedBoundingBox.h"
#include "Pathfinding.h"
#include "Plane.h"
#include "Shader.h"
#include "ScreenRect.h"
#include "Sphere.h"
#include "StaticOctTree.h"
#include "stbWrangler.h"
#include "Texture2D.h"
#include "UniformBuffer.h"
#include "util.h"
#include "Vertex.h"
#include "VertexArray.h"
#include "Wall.h"
#include "UserInterface.h"

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

static const std::array<GLubyte, 25> tesselationCode = { 
	{
		0xFF, 0x80, 0xFF, 0x20, 0x40,
		0x88, 0x00, 0x80, 0x43, 0xFC,
		0xFF, 0x80, 0xFF, 0xF0, 0xC0,
		0x80, 0x00, 0x8F, 0x90, 0xCF,
		0xD0, 0x80, 0xDF, 0xF0, 0xA0,
	}
};

ASCIIFont fonter;

// Buffers
Buffer<ArrayBuffer> albertBuffer, textBuffer, capsuleBuffer, instanceBuffer, plainCube, planeBO, rayBuffer, sphereBuffer, stickBuffer, texturedPlane;
Buffer<ArrayBuffer> cubeMesh, movingCapsule, normalMapBuffer;
Buffer<ElementArray> capsuleIndex, cubeOutlineIndex, movingCapsuleIndex, sphereIndicies, stickIndicies;

UniformBuffer cameraUniformBuffer, screenSpaceBuffer;

// Framebuffer
Framebuffer<2, DepthAndStencil> depthed;
Framebuffer<1, DepthStencil> toRemoveError;
ColorFrameBuffer scratchSpace;

// Shaders
Shader dither, expand, finalResult, flatLighting, fontShader, frameShader, ground, instancing, uiRect, uiRectTexture, uniform, sphereMesh, widget;

Shader stencilTest;

// Textures
Texture2D depthMap, ditherTexture, hatching, normalMap, tessMap, texture, wallTexture;
Texture2D buttonA, buttonB;
CubeMap mapper;

// Vertex Array Objects
VAO fontVAO, instanceVAO, meshVAO, normalVAO, normalMapVAO, plainVAO, texturedVAO;


// Not explicitly tied to OpenGL Globals
std::mutex bufferMutex;

OBB smartBox, dumbBox;
std::vector<Model> planes;
StaticOctTree<Dummy> boxes(glm::vec3(20));

static unsigned int frameCounter = 0;
bool smartBoxColor = false;

glm::vec3 moveSphere(0, 3.5f, 6.5f);
int kernel = 0;
int lineWidth = 3;

int windowWidth = 1000, windowHeight = 1000;
float aspectRatio = 1.f;
static const float Fov = 70.f;

#define TIGHT_BOXES 1
#define WIDE_BOXES 2
// One for each number key
std::array<bool, '9' - '0' + 1> debugFlags{};

// Input Shenanigans
#define ArrowKeyUp    0
#define ArrowKeyDown  1
#define ArrowKeyRight 2
#define ArrowKeyLeft  3

ColorFrameBuffer playerTextEntry;
std::stringstream letters("abc");
bool reRender = true;

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

bool buttonToggle = false;
ScreenRect buttonRect{ 540, 200, 100, 100 }, userPortion(0, 800, 1000, 200);

Capsule catapult;
Model catapultModel;
OBB catapultBox;

// TODO: Line Shader with width, all the math being on gpu (given the endpoints and the width then do the orthogonal to the screen kinda thing)
// TODO: Move cube stuff into a shader or something I don't know

OBB loom;

OBB moveable;

int tessAmount = 5;

bool featureToggle = false;
std::chrono::nanoseconds idleTime, displayTime;

constexpr float BulletRadius = 0.05f;

struct Bullet
{
	glm::vec3 position, direction;
};

std::vector<Bullet> bullets;
// TODO: Look into GLM_FORCE_INTRINSICS


/*
New shading outputs
-A stencil lights factor(mix(current, light, factor))
-

*/

void display()
{
	const auto now = std::chrono::high_resolution_clock::now();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	depthed.Bind();
	glViewport(0, 0, windowWidth, windowHeight);
	glClearColor(0, 0, 0, 1);
	// TODO: Maybe clear the stencil buffer explicitly? idk
	auto& sten = depthed.GetStencil();

	EnableGLFeatures<DepthTesting | FaceCulling>();
	glDepthMask(GL_TRUE);
	ClearFramebuffer<ColorBuffer | DepthBuffer | StencilBuffer>();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	DisableGLFeatures<StencilTesting>();
	/*
	EnableGLFeatures<StencilTesting>();
	glStencilFunc(GL_ALWAYS, 0x00, 0xFF);
	glStencilMask(0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
	*/
	// TODO: fix this thing so it's more efficient
	// Camera matrix
	glm::vec3 angles2 = glm::radians(cameraRotation);

	// Adding pi/2 is necessary because the default camera is facing -z
	glm::mat4 view = glm::translate(glm::eulerAngleXYZ(angles2.x, angles2.y + glm::half_pi<float>(), angles2.z), -cameraPosition);
	cameraUniformBuffer.BufferSubData(view, 0);

	DisableGLFeatures<Blending>();
	instancing.SetActiveShader();
	instancing.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	instancing.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	instancing.SetVec3("viewPos", cameraPosition);
	instancing.SetTextureUnit("textureIn", wallTexture, 0);
	instancing.SetTextureUnit("ditherMap", ditherTexture, 1);
	instancing.SetTextureUnit("normalMapIn", normalMap, 2);
	instancing.SetTextureUnit("depthMapIn", depthMap, 3);
	instancing.SetInt("flops", featureToggle);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//DisableGLFeatures<FaceCulling>();
	instanceVAO.BindArrayBuffer(texturedPlane, 0);
	instanceVAO.BindArrayBuffer(instanceBuffer, 1);
	instanceVAO.BindArrayBuffer(normalMapBuffer, 2);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei) planes.size());
	//EnableGLFeatures<Blending>();

	/* STICK FIGURE GUY */
	uniform.SetActiveShader();
	plainVAO.BindArrayBuffer(stickBuffer);

	glm::vec3 colors = glm::vec3(1, 0, 0);
	Model m22(glm::vec3(10, 0, 0));
	uniform.SetMat4("Model", m22.GetModelMatrix());
	uniform.SetVec3("color", colors);
	uniform.DrawIndexed<LineStrip>(stickIndicies);

	DisableGLFeatures<FaceCulling>();
	ground.SetActiveShader();
	glPatchParameteri(GL_PATCH_VERTICES, 4);
	texturedVAO.BindArrayBuffer(texturedPlane);
	ground.SetTextureUnit("heightMap", tessMap, 0);
	m22.translation = glm::vec3(1, 0, 0);
	m22.scale = glm::vec3(10);
	ground.SetMat4("Model", m22.GetModelMatrix());
	ground.SetInt("redLine", 0);
	ground.SetInt("amount", tessAmount);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//ground.DrawElements<Patches>(4);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	ground.SetInt("redLine", 1);
	//ground.DrawElements<Patches>(4);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	EnableGLFeatures<FaceCulling>();

	// Debugging boxes
	if (debugFlags[TIGHT_BOXES] || debugFlags[WIDE_BOXES])
	{
		uniform.SetActiveShader();
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
				//uniform.DrawIndexedMemory<Triangle>(cubeIndicies);
				uniform.DrawIndexed<Lines>(cubeOutlineIndex);
				uniform.DrawElements<Points>(8);
			}
			if (debugFlags[WIDE_BOXES])
			{
				uniform.SetVec3("color", (box.color) ? colors : blue);
				uniform.SetMat4("Model", box.box.GetAABB().GetModel().GetModelMatrix());
				uniform.DrawIndexed<Lines>(cubeOutlineIndex);
				//uniform.DrawElements<Points>(8);
			}
		}
	}

	// Cubert
	uniform.SetActiveShader();
	plainVAO.BindArrayBuffer(plainCube);
	uniform.SetMat4("Model", dumbBox.GetModelMatrix());
	
	glDepthMask(GL_TRUE);
	uniform.SetVec3("color", glm::vec3(1, 1, 1));
	uniform.SetMat4("Model", moveable.GetModelMatrix());
	uniform.DrawIndexedMemory<Triangle>(cubeIndicies);
	//glDepthMask(GL_FALSE)
	// Albert
	
	DisableGLFeatures<FaceCulling>();
	glPatchParameteri(GL_PATCH_VERTICES, 3);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	texturedVAO.BindArrayBuffer(albertBuffer);
	dither.SetActiveShader();
	dither.SetTextureUnit("ditherMap", wallTexture, 1);
	dither.SetTextureUnit("textureIn", texture, 0);
	dither.SetMat4("Model", smartBox.GetModelMatrix());
	dither.SetVec3("color", (!smartBoxColor) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0));
	dither.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	dither.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	dither.SetVec3("viewPos", cameraPosition);
	//dither.DrawElements<Patches>(albertBuffer);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	EnableGLFeatures<FaceCulling>();


	plainVAO.BindArrayBuffer(plainCube);
	uniform.SetActiveShader();
	uniform.SetMat4("Model", smartBox.GetModelMatrix());
	uniform.SetMat4("Model", catapult.GetAABB().GetModel().GetModelMatrix());
	//uniform.DrawIndexed<Lines>(cubeOutlineIndex);

	// Drawing of the rays
	//DisableGLFeatures<DepthTesting>();
	plainVAO.BindArrayBuffer(rayBuffer);
	Model bland;
	uniform.SetMat4("Model", bland.GetModelMatrix());
	glLineWidth(15.f);
	uniform.DrawElements<Lines>(rayBuffer);
	glLineWidth(1.f);
	//EnableGLFeatures<DepthTesting>();

	// Sphere drawing
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	flatLighting.SetActiveShader();

	normalVAO.BindArrayBuffer(sphereBuffer);

	Model sphereModel(glm::vec3(6.5f, 5.5f, 0.f));
	sphereModel.translation += glm::vec3(0, 1, 0) * glm::sin(glm::radians(frameCounter * 0.5f)) * 0.25f;
	sphereModel.scale = glm::vec3(1.5f);
	//sphereModel.rotation += glm::vec3(0.5f, 0.25, 0.125) * (float) frameCounter;
	//sphereModel.rotation += glm::vec3(0, 0.25, 0) * (float) frameCounter;
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
	for (auto& bullet : bullets)
	{
		Model localModel;
		localModel.translation = bullet.position;
		localModel.scale = glm::vec3(0.05f);
		sphereMesh.SetMat4("modelMat", localModel.GetModelMatrix());
		sphereMesh.SetMat4("normalMat", localModel.GetNormalMatrix());
		sphereMesh.SetTextureUnit("textureIn", texture, 0);
		//mapper.BindTexture(0);
		//sphereMesh.SetTextureUnit("textureIn", 0);
		sphereMesh.DrawIndexed<Triangle>(sphereIndicies);
	}


	sphereModel.scale = glm::vec3(4.f, 4.f, 4.f);
	sphereModel.translation = glm::vec3(0, 0, 0);
	Model lightModel;
	lightModel.translation = glm::vec3(4, 0, 0);
	lightModel.scale = glm::vec3(2.2, 2.2, 1.1);

	sphereModel.translation += glm::vec3(0, 1, 0) * glm::sin(glm::radians(frameCounter * 0.25f)) * 3.f;
	stencilTest.SetActiveShader();

	// TODO: Make something to clarify the weirdness of the stencil function
	// Stuff like the stencilOp being in order: Stencil Fail(depth ignored), Stencil Pass(Depth Fail), Stencil Pass(Depth Pass)
	// And stencilFunc(op, ref, mask) does the operation on a stencil value K of: (ref & mask) op (K & mask)

	// All shadows/lighting will be in this post-processing step based on stencil value
	// 


	// Useful only when view is *inside* the volume, should be reversed otherwise <- You are stupid

	//////  Shadow volume
	glDepthMask(GL_FALSE); // Disable writing to the depth buffer

	// Stencil tests must be active for stencil shading to be used
	// Depth testing is required to determine what faces the volumes intersect
	EnableGLFeatures<StencilTesting | DepthTesting>();
	// We are using both the front and back faces of the model, cannot be culling either
	DisableGLFeatures<FaceCulling>();

	// To make the inverse kind of volume (shadow/light), simply change the handedness of the system AND BE SURE TO CHANGE IT BACK
	//glFrontFace((featureToggle) ? GL_CCW : GL_CW);
	glFrontFace(GL_CCW);
	// Stencil Test Always Passes
	glStencilFunc(GL_ALWAYS, 0, 0xFF);
	
	// Back Faces increment the stencil value if they are behind the geometry, ie the geometry
	// is inside the volume
	glStencilOpSeparate(GL_BACK, GL_KEEP, GL_INCR_WRAP, GL_KEEP);
	// Front faces decrement if they are behind geometry, so that surfaces closer to the camera
	// than the volume are not incorrectly shaded by volumes that don't touch it
	glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_DECR_WRAP, GL_KEEP);

	// Somehow write the normal to the nearest light source? This feels like it should be simple but idk how to do it

	// Drawing of the appropriate volumes
	stencilTest.SetMat4("Model", sphereModel.GetModelMatrix());
	meshVAO.BindArrayBuffer(sphereBuffer);
	stencilTest.DrawIndexed<Triangle>(sphereIndicies);

	plainVAO.BindArrayBuffer(plainCube);
	stencilTest.SetMat4("Model", lightModel.GetModelMatrix());
	stencilTest.DrawIndexedMemory<Triangle>(cubeIndicies);

	// Clean up
	EnableGLFeatures<FaceCulling>();
	//DisableGLFeatures<StencilTesting>();
	glFrontFace(GL_CCW);
	//glDepthMask(GL_TRUE); // Allow for the depth buffer to be written to
	//////  Shadow volume End

	//GL_ARB_shader_stencil_export

	EnableGLFeatures<Blending>();
	//DisableGLFeatures<DepthTesting>();
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glStencilFunc(GL_LEQUAL, 1, 0xFF); // If 1 is <= value in the stencil buffer the test passes

	glStencilFunc(GL_GEQUAL, 1, 0xFF); // If 1 is >= value in the stencil buffer the test passes
	//glStencilOpSeparate(GL_FRONT_AND_BACK, GL_KEEP, GL_KEEP, GL_KEEP);
	glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
	//glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	uiRect.SetActiveShader();
	uiRect.SetVec4("color", glm::vec4(0, 0, 0, 0.8));
	uiRect.SetVec4("rectangle", glm::vec4(0, 0, windowWidth, windowHeight));
	//uiRect.DrawElements(TriangleStrip, 4);
	//uiRect.DrawElements(TriangleStrip, 4);

	DisableGLFeatures<StencilTesting>();
	//EnableGLFeatures<DepthTesting>();
	glDepthMask(GL_TRUE);

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


	meshVAO.BindArrayBuffer(movingCapsule);
	flatLighting.SetMat4("modelMat", catapultBox.GetNormalMatrix());
	flatLighting.SetMat4("normalMat", catapultBox.GetNormalMatrix());
	flatLighting.DrawIndexed<Triangle>(movingCapsuleIndex);
	// Calling with triangle_strip is fucky
	/*
	flatLighting.DrawIndexed(Triangle, sphereIndicies);
	sphereModel.translation = moveSphere;
	flatLighting.SetMat4("modelMat", sphereModel.GetModelMatrix());
	flatLighting.SetMat4("normMat", sphereModel.GetNormalMatrix());
	flatLighting.DrawIndexed(Triangle, sphereIndicies);
	*/

	DisableGLFeatures<DepthTesting>();
	EnableGLFeatures<Blending>();
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	uiRect.SetActiveShader();
	uiRect.SetVec4("color", glm::vec4(0, 0.5, 0.75, 0.25));
	uiRect.SetVec4("rectangle", glm::vec4(0, 0, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);
	
	uiRect.SetVec4("rectangle", glm::vec4(windowWidth - 200, 0, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);
	
	uiRect.SetVec4("rectangle", glm::vec4(0, windowHeight - 100, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);
	
	uiRect.SetVec4("rectangle", glm::vec4(windowWidth - 200, windowHeight - 100, 200, 100));
	uiRect.DrawElements(TriangleStrip, 4);

	uiRect.SetVec4("rectangle", userPortion);
	uiRect.SetVec4("color", glm::vec4(0.25, 0.25, 0.25, 0.85));
	//uiRect.DrawElements(TriangleStrip, 4);

	uiRectTexture.SetActiveShader();
	auto& colored = playerTextEntry.GetColor();
	uiRectTexture.SetTextureUnit("image", colored, 0);
	uiRectTexture.SetVec4("rectangle", glm::vec4((windowWidth - colored.GetWidth()) / 2, (windowHeight - colored.GetHeight()) / 2, 
		colored.GetWidth(), colored.GetHeight()));
	//uiRect.DrawElements(TriangleStrip, 4);

	uiRectTexture.SetTextureUnit("image", (buttonToggle) ? buttonA : buttonB, 0);
	uiRectTexture.SetVec4("rectangle", buttonRect);
	uiRect.DrawElements(TriangleStrip, 4);

	// Debug Info Display
	fontShader.SetActiveShader();
	fontVAO.BindArrayBuffer(textBuffer);
	fontShader.SetTextureUnit("fontTexture", fonter.GetTexture(), 0);
	fontShader.DrawElements<Triangle>(textBuffer);
	// TODO: Set object amount in buffer function

	DisableGLFeatures<Blending>();
	EnableGLFeatures<DepthTesting>();

	// Framebuffer stuff
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	//scratchSpace.Bind();
	//glClearColor(0, 0, 0, 1);
	//glClear(GL_COLOR_BUFFER_BIT);

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
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	DisableGLFeatures<DepthTesting | StencilTesting | FaceCulling>();

	glStencilMask(0x00);
	glStencilFunc(GL_ALWAYS, GL_KEEP, GL_KEEP);

	expand.SetActiveShader();
	expand.SetTextureUnit("screen", depthed.GetColorBuffer<0>(), 0);
	expand.SetTextureUnit("edges", depthed.GetColorBuffer<1>(), 1);
	expand.SetTextureUnit("depths", depthed.GetDepth(), 2);
	expand.SetTextureUnit("stencil", depthed.GetStencil(), 3);
	expand.SetInt("depth", 5);
	expand.SetInt("flag", featureToggle);
	frameShader.DrawElements<TriangleStrip>(4);
	glStencilMask(0xFF);

	glLineWidth(1.f);
	widget.SetActiveShader();
	widget.DrawElements<Lines>(6);

	EnableGLFeatures<DepthTesting | StencilTesting | FaceCulling>();

	// TODO: Seperate cpu rendering and render latency timers
	//glFlush();
	glFinish();
	auto end = std::chrono::high_resolution_clock::now();
	displayTime = end - now;
}

static const glm::vec3 GravityAxis{ 0.f, -1.f, 0.f };
static const glm::vec3 GravityUp{ 0.f, 1.f, 0.f };

struct
{
	glm::vec3 acceleration{ 0.f };
	glm::vec3 velocity{ 0.f };
	glm::vec3 axisOfGaming{ 0.f };
	OBB* ptr = nullptr;
} smartBoxPhysics;

static float maxRotatePerFrame = 0.f;

// Aligns to the corner
void smartBoxAlignCorner(OBB& other, glm::length_t minDotI, glm::length_t maxDotI)
{
	// Only the scale of the other one is needed to determine if this midpoint is inside
	glm::vec3 dumbScale = other.GetScale();
	glm::vec3 delta = other.Center() - smartBox.Center();

	// Maintain right handedness
	// TODO: maybe 2, 0 for the second one?
	int indexA[3] = { 0, 0, 1 };
	int indexB[3] = { 1, 2, 2 };
	for (int i = 0; i < 3; i++)
	{
		int indexedA = indexA[i];
		int indexedB = indexB[i];

		glm::vec3 axisA = other[indexedA];
		glm::vec3 axisB = other[indexedB];

		// Calculate the distance along each of these axes 
		float projectionA = glm::dot(delta, axisA);
		float projectionB = glm::dot(delta, axisB);

		// See if the extent in that direction is more than covered by these axes
		bool testA = glm::abs(projectionA) >= dumbScale[indexedA];
		bool testB = glm::abs(projectionB) >= dumbScale[indexedB];

		// smartBox collides entirely because of its own sides, therefore it might need to rotate
		if (testA && testB)
		{
			glm::quat current = glm::quat_cast(smartBox.GetNormalMatrix());
			bool axisTest = glm::abs(projectionA) > glm::abs(projectionB);

			// This is the axis of smartBox that will be rotated towards?
			glm::vec3 localAxis = (axisTest) ? axisA : axisB;
			if (glm::sign(glm::dot(localAxis, delta)) > 0)
			{
				localAxis *= -1;
			}
			glm::vec3 rotationAxis = glm::normalize(glm::cross(smartBox[maxDotI], localAxis));

			glm::quat rotation = glm::angleAxis(glm::acos(glm::dot(smartBox[maxDotI], localAxis)), rotationAxis);
			glm::quat newer = glm::normalize(rotation * current);
			glm::quat older = glm::normalize(current);

			float maxDelta = glm::acos(glm::abs(glm::dot(older, newer)));
			float clamped = std::clamp(maxDelta, 0.f, glm::min(1.f, maxRotatePerFrame));
			if (glm::abs(glm::dot(older, newer) - 1) > EPSILON)
			{
				// Slerp interpolates along the shortest axis on the great circle
				smartBox.ReOrient(glm::toMat4(glm::normalize(glm::slerp(older, newer, clamped))));
			}
			break;
		}
	}
}

// Aligns to the Face
void smartBoxAlignFace(OBB& other, glm::vec3 axis, glm::length_t minDotI, glm::length_t maxDotI)
{
	glm::mat3 goobers{ smartBox[0], smartBox[1], smartBox[2] };
	// Leasted aligned keeps its index
	// Middle is replaced with least cross intersection
	// Most is replaced with the negative of new middle cross least
	glm::vec3 least = smartBox[minDotI];                           // goes in smartbox[minDotI]
	glm::vec3 newMost = axis;                              // goes in smartbox[maxDotI]
	glm::vec3 newest = glm::normalize(glm::cross(least, newMost)); // goes in the remaining one(smartbox[3 - minDotI - maxDotI])

	//std::cout << minDotI << ":" << maxDotI << std::endl;

	glm::length_t leastD = minDotI;
	glm::length_t mostD = maxDotI;
	glm::length_t newD = 3 - leastD - mostD;
	if (newD != 3)
	{
		least *= glm::sign(glm::dot(least, goobers[minDotI]));
		newMost *= glm::sign(glm::dot(newMost, goobers[maxDotI]));
		newest *= glm::sign(glm::dot(newest, goobers[newD]));

		glm::mat3 lame{};
		lame[leastD] = least;
		lame[mostD] = newMost;
		lame[newD] = newest;
		glm::quat older = glm::normalize(glm::quat_cast(smartBox.GetNormalMatrix()));
		glm::quat newer = glm::normalize(glm::quat_cast(lame));
		float maxDelta = glm::acos(glm::abs(glm::dot(older, newer)));
		float clamped = std::clamp(maxDelta, 0.f, glm::min(1.f, maxRotatePerFrame));

		// ?
		//if (glm::abs(glm::acos(glm::dot(older, newer))) > EPSILON)
		if (glm::abs(glm::dot(older, newer) - 1) > EPSILON)
		{
			// Slerp interpolates along the shortest axis on the great circle
			smartBox.ReOrient(glm::toMat4(glm::normalize(glm::slerp(older, newer, clamped))));
		}
	}
	else
	{
		std::cout << "Something went horribly wrong" << std::endl;
	}
}

void smartBoxAligner(OBB& other, glm::vec3 axis)
{
	float minDot = INFINITY, maxDot = -INFINITY;
	glm::length_t minDotI = 0, maxDotI = 0;
	for (glm::length_t i = 0; i < 3; i++)
	{
		float local = glm::abs(glm::dot(smartBox[i], axis));
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
	}
	smartBoxAlignFace(other, axis, minDotI, maxDotI);
}


bool smartBoxCollide()
{
	bool anyCollisions = false;
	smartBoxPhysics.axisOfGaming = glm::vec3{ 0.f };
	smartBoxPhysics.ptr = nullptr;

	auto potentialCollisions = boxes.Search(smartBox.GetAABB());
	int collides = 0;
	//Before(smartBoxPhysics.axisOfGaming);
	float dot = INFINITY;

	// TODO: Another test with a slightly scaled up version if there is no intersection, to provide a surface normal or a rotation target
	for (auto& currentBox : potentialCollisions)
	{
		SlidingCollision c;
		if (smartBox.Overlap(currentBox->box, c))
		{
			float temp = glm::dot(c.axis, GravityUp);
			//smartBoxPhysics.velocity += c.axis * c.depth * 0.95f;
			//smartBoxPhysics.velocity -= c.axis * glm::dot(c.axis, smartBoxPhysics.velocity);
			if (temp > 0 && temp <= dot)
			{
				temp = dot;
				smartBoxPhysics.axisOfGaming = c.axis;
				smartBoxPhysics.ptr = &currentBox->box;
			}

			float minDot = INFINITY, maxDot = -INFINITY;
			glm::length_t minDotI = 0, maxDotI = 0;
			glm::vec3 upper = c.axis;
			for (glm::length_t i = 0; i < 3; i++)
			{
				float local = glm::abs(glm::dot(smartBox[i], upper));
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
			}

			smartBoxAlignCorner(currentBox->box, minDotI, maxDotI);

			// TODO: look into this
			//if (glm::acos(glm::abs(maxDotI - 1)) > EPSILON)
			// TODO: Thing to make sure this isn't applied when the box itself is rotating under its own power
			//if (true) //(c.depth > 0.002) // Why this number
			if (!(keyState[ArrowKeyRight] || keyState[ArrowKeyLeft]))
			{
				smartBoxAlignFace(currentBox->box, upper, minDotI, maxDotI);
				smartBox.OverlapAndSlide(currentBox->box);
			}
			else
			{ 
				smartBox.ApplyCollision(c);
			}
			float oldLength = glm::length(smartBoxPhysics.velocity);
			anyCollisions = true;
		}
	}
	
	// Scale smart box up a bit to determine axis and
	if (!anyCollisions) 
	{
		// This is probably a bad idea
		
		glm::vec3 oldCenter = smartBox.Center();
		smartBox.Translate(GravityAxis * 2.f * EPSILON);

		potentialCollisions = boxes.Search(smartBox.GetAABB());
		SlidingCollision newest{};
		OBB* newPtr = nullptr;

		for (auto& currentBox : potentialCollisions)
		{
			SlidingCollision c;
			if (smartBox.Overlap(currentBox->box, c))
			{
				float temp = glm::dot(c.axis, GravityUp);
				if (temp > 0 && temp <= dot)
				{
					temp = dot;
					smartBoxPhysics.axisOfGaming = c.axis;
					smartBoxPhysics.ptr = &currentBox->box;
					newest = c;
					newPtr = &currentBox->box;
				}
			}
		}
		smartBox.ReCenter(oldCenter);
		if (newPtr)
		{
			OBB& box = *newPtr;
			float minDot = INFINITY, maxDot = -INFINITY;
			glm::length_t minDotI = 0, maxDotI = 0;
			glm::vec3 upper = newest.axis;
			for (glm::length_t i = 0; i < 3; i++)
			{
				float local = glm::abs(glm::dot(smartBox[i], upper));
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
			}
			smartBoxAlignCorner(box, minDotI, maxDotI);
			smartBoxAlignFace(box, upper, minDotI, maxDotI);
		}
	}
	
	return anyCollisions;
}

glm::quat oldMan{};
glm::quat newMan{};

OBB* capsuleHit;
glm::vec3 capsuleNormal, capsuleAcceleration, capsuleVelocity;


// TODO: Mech suit has an interior for the pilot that articulates seperately from the main body, within the outer limits of the frame
// Like it's a bit pliable
void idle()
{
	static auto lastTimers = std::chrono::high_resolution_clock::now();
	static std::deque<float> frames;
	static std::deque<long long> displayTimes, idleTimes;
	static glm::vec3 previously{};


	frameCounter++;
	const auto now = std::chrono::high_resolution_clock::now();
	const auto delta = now - lastTimers;

	OBB goober2(AABB(glm::vec3(0), glm::vec3(1)));
	goober2.Translate(glm::vec3(2, 0.1, 0));	
	goober2.Rotate(glm::radians(glm::vec3(0, frameCounter * 4.f, 0)));
	glm::mat4 tester = goober2.GetNormalMatrix();

	Plane foobar(glm::vec3(1, 0, 0), glm::vec3(4, 0, 0)); // Facing away from origin
	//foobar.ToggleTwoSided();
	//if (!smartBox.IntersectionWithResponse(foobar))
		//sacounter++;
		//std::cout << counter << std::endl;
	//smartBox.RotateAbout(glm::vec3(0.05f, 0.07f, -0.09f), glm::vec3(0, -5, 0));
	//smartBox.RotateAbout(glm::vec3(0, 0, 0.05f), glm::vec3(0, -2, 0));
	//smartBox.RotateAbout(glm::vec3(0.05f, 0, 0), glm::vec3(0, -5, 0));

	const float timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(delta).count();
	
	// TODO: Rolling buffer size thingy setting
	auto idleDelta = idleTime.count() / 1000;
	auto displayDelta = displayTime.count() / 1000;
	frames.push_back(1.f / timeDelta);
	displayTimes.push_back(displayDelta);
	idleTimes.push_back(idleDelta);
	if (frames.size() > 300)
	{
		frames.pop_front();
		displayTimes.pop_front();
		idleTimes.pop_front();
	}
	float averageFps = 0.f;
	long long averageIdle = 0, averageDisplay = 0;
	for (std::size_t i = 0; i < frames.size(); i++)
	{
		averageFps += frames[i] / frames.size();
		averageDisplay += displayTimes[i] / displayTimes.size();
		averageIdle += idleTimes[i] / idleTimes.size();
	}
	// TODO: Consider using an expoentially weighted average, average = x * current + (1 - x) * previous_average to save time on addition stuff

	previously = smartBoxPhysics.axisOfGaming;

	/*
	fonter.RenderToScreen(textBuffer, 0, 0, std::format("Right: {}\nLeft: {}\nUp: {}\nDown: {}", keyState[ArrowKeyRight], keyState[ArrowKeyLeft],
		keyState[ArrowKeyUp], keyState[ArrowKeyDown]));
	*/
	// End of Rolling buffer

	float speed = 4 * timeDelta;
	float turnSpeed = 100 * timeDelta;
	maxRotatePerFrame = glm::radians(90.f) * timeDelta;

	glm::vec3 forward = glm::eulerAngleY(glm::radians(-cameraRotation.y)) * glm::vec4(1, 0, 0, 0);
	glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
	forward = speed * glm::normalize(forward);
	right = speed * glm::normalize(right);
	glm::vec3 previous = cameraPosition;
	if (keyState['Z'])
	{
		dumbBox.Translate(dumbBox.Forward() * speed);
	}
	if (keyState['X'])
	{
		dumbBox.Translate(dumbBox.Forward() * -speed);
	}
	if (keyState[ArrowKeyRight]) smartBox.Rotate(glm::vec3(0, -1.f, 0) * turnSpeed);
	if (keyState[ArrowKeyLeft])  smartBox.Rotate(glm::vec3(0, 1.f, 0) * turnSpeed);

	if (keyState[ArrowKeyRight]) catapultBox.Rotate(glm::vec3(0, -1.f, 0) * turnSpeed);
	if (keyState[ArrowKeyLeft])  catapultBox.Rotate(glm::vec3(0, 1.f, 0) * turnSpeed);
	if (keyState['W'])
		cameraPosition += forward;
	if (keyState['S'])
		cameraPosition -= forward;
	if (keyState['D'])
		cameraPosition += right;
	if (keyState['A'])
		cameraPosition -= right;
	if (keyState['Z']) cameraPosition += GravityUp * speed;
	if (keyState['X']) cameraPosition -= GravityUp * speed;
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
	const float BoxAcceleration = 0.06f;
	const float BoxMass = 1;
	const float staticFrictionCoeff = 1.0f;
	const float slidingFrictionCoeff = 0.57f;
	const float GravityRate = 0.25f;
	const float BoxGravityMagnitude = BoxMass * GravityRate;
	const glm::vec3 boxGravity = GravityAxis * BoxGravityMagnitude;
	const float cose = glm::abs(glm::dot(smartBoxPhysics.axisOfGaming, GravityUp));
	const float sine = glm::sqrt(1.f - cose * cose);
	const glm::vec3 oldNormal = smartBoxPhysics.axisOfGaming;

	glm::vec3 boxForces{ 0.f };

	//if (keyState[ArrowKeyUp])   boxForces += smartBox.Forward() * BoxAcceleration;
	//if (keyState[ArrowKeyUp])   smartBox.Translate(smartBox.Forward() * speed);
	//if (keyState[ArrowKeyDown]) boxForces -= smartBox.Forward() * BoxAcceleration;
	//if (keyState[ArrowKeyDown]) smartBox.Translate(-smartBox.Forward() * speed);
	if (keyState['V']) smartBox.Translate(GravityAxis * speed);
	if (keyState['C']) smartBox.Translate(GravityUp * speed);
	if (keyState['F']) smartBox.Translate(glm::vec3(0, 1, 0));
	if (keyState['G']) smartBox.Translate(-smartBoxPhysics.axisOfGaming * speed);

	if (frameCounter % 2000 == 0)
	{
		std::swap(oldMan, newMan);
	}
	unsigned int modded = frameCounter % 2000;
	//smartBox.ReOrient(glm::toMat4(glm::lerp(oldMan, newMan, modded / 2000.f)));
	
	float forwardDirection = float(keyState[ArrowKeyUp] ^ keyState[ArrowKeyDown]) * ((keyState[ArrowKeyDown]) ? -1.f : 1.f);
	bool groundFound = smartBoxPhysics.ptr != nullptr;
	if (!groundFound)
	{
		boxForces += boxGravity;
		boxForces += forwardDirection * forward * BoxAcceleration;
		//std::cout << "no ground :(" << std::endl;
	}
	else
	{
		float speedCoefficient = static_cast<float>(glm::sqrt(1 - glm::pow(glm::dot(smartBox.Forward(), smartBoxPhysics.axisOfGaming), 2)));
		boxForces += smartBox.Forward() * forwardDirection * speedCoefficient * BoxAcceleration;
	}

	// F = MA, -> A = F / M
	//std::cout << boxForces << std::endl;
	smartBoxPhysics.acceleration = boxForces / BoxMass * timeDelta;

	smartBoxPhysics.velocity += smartBoxPhysics.acceleration;
	if (glm::length(smartBoxPhysics.velocity) > 2.f)
		smartBoxPhysics.velocity = glm::normalize(smartBoxPhysics.velocity) * 2.f;
	smartBox.Translate(smartBoxPhysics.velocity);
	smartBoxPhysics.velocity *= 0.99f;

	smartBoxColor = smartBoxCollide();
	if (staticFrictionCoeff >= glm::tan(glm::acos(cose)))
	{
		//smartBoxPhysics.axisOfGaming = oldNormal;
	}

	//smartBox.OverlapCompleteResponse(dumbBox);

	// CAPSULE STUFF
	float mult = float(keyState[ArrowKeyUp] ^ keyState[ArrowKeyDown]) * ((keyState[ArrowKeyDown]) ? -1.f : 1.f);
	float capsuleDot = -INFINITY;
	glm::vec3 capsuleForces{};
	// Transformations need to be addressed
	if (!capsuleHit)
	{
		capsuleForces += boxGravity;
	}
	capsuleForces += catapultBox.Forward() * mult * BoxAcceleration;
	capsuleHit = nullptr;
	capsuleNormal = glm::vec3(0);
	capsuleAcceleration = capsuleForces / BoxMass * timeDelta;
	capsuleVelocity += capsuleAcceleration;
	//std::cout << catapult.GetCenter() << std::endl;
	if (glm::length(capsuleVelocity) > 2.f)
		capsuleVelocity = glm::normalize(capsuleVelocity) * 2.f;
	catapult.Translate(capsuleVelocity);
	capsuleVelocity *= 0.99f; // Maybe "real" friction?
	for (auto& temps : boxes.Search(catapult.GetAABB()))
	{
		Collision c;
		if (temps->box.Overlap(catapult, c))
		{
			catapult.Translate(c.normal * c.depth);
			float dot = glm::dot(GravityUp, c.normal);
			if (dot > 0 && dot > capsuleDot)
			{
				capsuleHit = &temps->box;
				capsuleDot = dot;
			}
		}
	}
	catapultBox.ReCenter(catapult.GetCenter());


	Sphere awwYeah(0.5f, moveSphere);
	for (auto& letsgo : boxes.Search(AABB(awwYeah.center - glm::vec3(awwYeah.radius), awwYeah.center + glm::vec3(awwYeah.radius))))
	{
		Collision c;
		if (letsgo->box.Overlap(awwYeah, c))
		{
			awwYeah.center += c.normal * c.depth;
		}
	}
	moveSphere = awwYeah.center;
	
	if (reRender && letters.str().size() > 0)
	{
		reRender = false;
		playerTextEntry = fonter.Render(letters.str(), glm::vec4(1, 0, 0, 1));
		std::stringstream().swap(letters);
	}
	fonter.RenderToScreen(textBuffer, 0, 0, std::format("FPS:{:7.2f}\nTime:{:4.2f}ms\nCPU:{}ns\nGPU:{}ns\n{} Version\nTest Bool: {}",
		averageFps, 1000.f / averageFps, averageIdle, averageDisplay, (featureToggle) ? "New" : "Old", capsuleHit == nullptr));

	const float BulletSpeed = 5.f * timeDelta; //  5 units per second
	Sphere gamin{};
	Collision c;
	gamin.radius = BulletRadius;
	for (std::size_t i = 0; i < bullets.size(); i++)
	{
		if (glm::any(glm::greaterThan(glm::abs(bullets[i].position), glm::vec3(20))))
		{
			bullets.erase(bullets.begin() + i);
			i--;
			continue;
		}
		gamin.center = bullets[i].position + bullets[i].direction * BulletSpeed;
		for (auto& boxers : boxes.Search(gamin.GetAABB()))
		{
			if (boxers->box.Overlap(gamin, c))
			{
				gamin.center = c.point;
				bullets[i].direction = glm::reflect(bullets[i].direction, c.normal);
			}
		}
		bullets[i].position = gamin.center;
	}

	std::copy(std::begin(keyState), std::end(keyState), std::begin(keyStateBackup));
	std::swap(keyState, keyStateBackup);

	const auto endTime = std::chrono::high_resolution_clock::now();
	idleTime = endTime - now;
	lastTimers = now;
	// Delay to keep 100 ticks per second idle stuff
	/*
	if (idleTime < std::chrono::milliseconds(10))
	{
		while (std::chrono::high_resolution_clock::now() - now <= std::chrono::milliseconds(10));
	}
	*/
}

void smartReset()
{
	smartBox.ReCenter(glm::vec3(12.2f, 1.6f, 0));
	smartBox.ReOrient(glm::vec3(0, 0, 0));
	/*
	smartBox.ReCenter(glm::vec3(2, 1.f, 0));
	smartBox.ReOrient(glm::vec3(0, 135, 0));
	*/
}

void window_focus_callback(GLFWwindow* window, int focused)
{
	if (!focused)
	{
		keyStateBackup.fill(false);
	}
}

void key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, int mods)
{
	bool state = action == GLFW_PRESS;

	unsigned char letter = (unsigned char)(key & 0xFF);

	if (action != GLFW_RELEASE && key < 0xFF)
	{
		letters << letter;
	}

	// If key is an ascii, then GLFW_KEY_* will be equal to '*', ie GLFW_KEY_M = 'M', all uppercase by default
	if (action != GLFW_REPEAT && key < 0xFF)
	{
		keyState[letter] = state;
	}
	if (action != GLFW_REPEAT && key > 0xFF)
	{
		switch (key)
		{
		case GLFW_KEY_UP: { keyState[ArrowKeyUp] = state; break; }
		case GLFW_KEY_DOWN: { keyState[ArrowKeyDown] = state; break; }
		case GLFW_KEY_RIGHT: { keyState[ArrowKeyRight] = state; break; }
		case GLFW_KEY_LEFT: { keyState[ArrowKeyLeft] = state; break; }
		default: break;
		}
	}

	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_M) cameraPosition.y += 3;
		if (key == GLFW_KEY_N) cameraPosition.y -= 3;
		if (key == GLFW_KEY_LEFT_BRACKET) tessAmount -= 1;
		if (key == GLFW_KEY_RIGHT_BRACKET) tessAmount += 1;
		if (key == 'Q') glfwSetWindowShouldClose(window, GLFW_TRUE);
		if (key == GLFW_KEY_F) kernel = 1 - kernel;
		if (key == GLFW_KEY_B) featureToggle = !featureToggle;
		if (key == GLFW_KEY_ENTER) reRender = true;
		if (key == 'H')
		{
			smartReset();
		}
		if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
		{
			std::size_t value = (std::size_t) key - GLFW_KEY_0;
			debugFlags[value] = !debugFlags[value];
		}
		if (key >= GLFW_KEY_F1 && key <= GLFW_KEY_F1 + debugFlags.size())
		{
			std::size_t value = (std::size_t)key - GLFW_KEY_F1;
			debugFlags[value] = !debugFlags[value];
		}
	}
}

// TODO: struct or something idk
static bool rightMouseHeld = false, leftMouseHeld = false;
static float mousePreviousX = 0, mousePreviousY = 0;

// TODO: this needs to be it's own function
glm::vec2 GetProjectionHalfs(glm::mat4& mat)
{
	glm::vec2 result{};

	Plane rightPlane(mat[0][3] - mat[0][0], mat[1][3] - mat[1][0], mat[2][3] - mat[2][0], -mat[3][3] + mat[3][0]);
	Plane topPlane  (mat[0][3] - mat[0][1], mat[1][3] - mat[1][1], mat[2][3] - mat[2][1], -mat[3][3] + mat[3][1]);
	Plane nearPlane (mat[0][3] + mat[0][2], mat[1][3] + mat[1][2], mat[2][3] + mat[2][2], -mat[3][3] - mat[3][2]);
	glm::vec3 local{};
	nearPlane.TripleIntersect(rightPlane, topPlane, local);
	result.x = local.x;
	result.y = local.y;
	return result;
}

Ray GetMouseProjection(const glm::vec2& mouse, glm::mat4& cameraOrientation)
{
	/* STEPS FROM GODOT: https://github.com/godotengine/godot/blob/80de898d721f952dac0b102d48bb73d6b02ee1e8/scene/3d/camera_3d.cpp#L390
	> Get Viewport size
	> Get camera projection, zNear being defined by the depth you want it to be
	> Get the half lengths of the camera projection
	> Given the input (x,y) coordinates compute
	> newX = (x / size.x) * 2.0 - 1.0
	> newY = (1.0 - (y / size.y)) * 2.0 - 1.0
	> (newX, newY) *= half lengths
	> Proejctioned vector, called p = (newX, newY, -depth)
	> Get the camera transform(?) then apply the function(below) to p
	newVec = (dot(basis[0], p) + originX, dot(basis[1], p) + originY, dot(basis[2], p) + originZ)
	*/
	float x = mouse.x, y = mouse.y;
	glm::vec2 viewPortSize{ windowWidth, windowHeight };
	glm::vec2 sizes((x / viewPortSize.x) * 2.0f - 1.0f, (1.0f - (y / viewPortSize.y)) * 2.0f - 1.0f);

	// Lets have depth = 0.01;
	float depth = 0.01f;
	glm::mat4 projection = glm::perspective(glm::radians(Fov), aspectRatio, depth, zFar);
	sizes *= GetProjectionHalfs(projection);
	glm::vec3 project(sizes.x, sizes.y, -depth);

	glm::vec3 radians = glm::radians(cameraRotation);

	// Center of screen orientation
	cameraOrientation = glm::eulerAngleXYZ(radians.x, radians.y + glm::half_pi<float>(), radians.z);

	glm::vec3 faced{ 0.f };
	for (int i = 0; i < 3; i++)
	{
		faced[i] = glm::dot(glm::vec3(cameraOrientation[i]), project);
		// To do a proper projection you would add the camera position but that isn't necessary for this use
	}
	faced = glm::normalize(faced);

	glm::vec3 axial = glm::normalize(glm::cross(glm::vec3(1, 0, 0), faced));
	float dist = glm::acos(glm::dot(glm::vec3(1, 0, 0), faced));

	// Orientation of the ray being shot
	cameraOrientation = glm::mat4_cast(glm::normalize(glm::angleAxis(dist, axial)));

	return Ray(cameraPosition, faced);
}

MouseStatus mouseStatus{};

void ButtonExample(std::size_t id)
{
	buttonToggle = !buttonToggle;
}

Button testButton{ buttonRect, ButtonExample };
Context contextul;

void mouseButtonFunc(GLFWwindow* window, int button, int action, int status)
{
	contextul.AddButton<BasicButton>(buttonRect, ButtonExample);
	// Set bit (button) in mouseStatus.buttons
	mouseStatus.buttons = (mouseStatus.buttons & ~(1 << button) | (action == GLFW_PRESS) << button);
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		rightMouseHeld = (action == GLFW_PRESS);
		if (rightMouseHeld)
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		}
		else
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT)
	{
		leftMouseHeld = (action == GLFW_PRESS);
	}

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !userPortion.Contains(mousePreviousX, mousePreviousY))
	{
		glm::mat4 cameraOrientation{};
		Ray liota = GetMouseProjection(glm::vec2(mousePreviousX, mousePreviousY), cameraOrientation);
		float rayLength = 50.f;

		RayCollision rayd{};
		Dummy* point = nullptr;
		for (auto& item : boxes.RayCast(liota))
		{
			if (item->box.Intersect(liota.initial, liota.delta, rayd) && rayd.depth > 0.f && rayd.depth < rayLength)
			{
				rayLength = rayd.depth;
				point = &(*item);
			}
		}
		// Point now has the pointer to the closest element
		Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.1f, rayLength - 0.5f - 0.2f, 30, 30);
		//loom.ReOrient(glm::vec3(0, 0, 90.f));
		loom.ReOrient(cameraOrientation);
		//loom.ReCenter(cameraPosition);
		//loom.Translate(loom.Forward() * (0.3f + rayLength / 2.f));
		loom.Rotate(glm::vec3(0, 0, 90.f));
		loom.ReScale(glm::vec3((rayLength - 0.5f) / 2.f, 0.1f, 0.1f));
		Bullet locals{};
		locals.position = cameraPosition;
		locals.direction = liota.delta;
		bullets.push_back(locals);
	}
	testButton.MouseUpdate(mouseStatus);
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && userPortion.Contains(mousePreviousX, mousePreviousY))
	{
		userPortion.z -= 25;
	}
}

void mouseCursorFunc(GLFWwindow* window, double xPos, double yPos)
{
	float x = static_cast<float>(xPos), y = static_cast<float>(yPos);
	mouseStatus.position = glm::vec2(x, y);
	if (rightMouseHeld)
	{
		float xDif = x - mousePreviousX;
		float yDif = y - mousePreviousY;
		if (abs(xDif) > 20)
			xDif = 0;
		if (abs(yDif) > 20)
			yDif = 0;
		// Why 50??
		float yDelta = 50 * (xDif * ANGLE_DELTA) / windowWidth;
		float zDelta = 50 * (yDif * ANGLE_DELTA) / windowHeight;

		cameraRotation.y += yDelta;
		cameraRotation.x = std::clamp(cameraRotation.x + zDelta, -75.f, 75.f);
	}
	else
	{
		//buttonToggle = buttonRect.Contains(x, y);
	}
	if (leftMouseHeld)
	{
		glm::mat4 __unused{};
		Ray liota(GetMouseProjection(glm::vec2(x, y), __unused));
		RayCollision rayd{};
		if (moveable.Intersect(liota.initial, liota.delta, rayd) && rayd.depth > 0)
		{
			// Maybe take derivative of the circle the view makes?
			// Could just do the simple thing of "attaching" the object to the mouse then moving or something idfk this is stupid
			glm::length_t hitSide = 0;
			for (glm::length_t i = 0; i < 3; i++)
			{
				if (rayd.normal == moveable[i])
				{
					hitSide = i;
					break;
				}
			}
			glm::length_t otherA, otherB;
			if (hitSide == 0) { otherA = 1; otherB = 2; }
			if (hitSide == 1) { otherA = 0; otherB = 2; }
			if (hitSide == 2) { otherA = 0; otherB = 1; }

			// We Hit it!
			//I still don't understand why it has to be like this
			glm::vec3 radians = -glm::radians(cameraRotation);
			glm::mat4 cameraOrientation = glm::eulerAngleXYZ(radians.z, radians.y, radians.x);
			float xDif = x - mousePreviousX;
			float yDif = y - mousePreviousY;
			// Why 50??
			float yDelta = (xDif * ANGLE_DELTA) / windowWidth;
			float zDelta = -(yDif * ANGLE_DELTA) / windowHeight;


			// Temporarily turned backwards
			glm::vec3 camUp = glm::vec3(cameraOrientation[2]), camOrth = glm::vec3(cameraOrientation[0]);
			glm::vec3 targetAxisA = moveable[otherA], targetAxisB = moveable[otherB];



			float minDot = -INFINITY, maxDot = -INFINITY;
			glm::length_t minDotI = 0, maxDotI = 0;
			for (glm::length_t i = 0; i < 3; i++)
			{
				float local = glm::abs(glm::dot(moveable[i], camUp));
				float local2 = glm::abs(glm::dot(moveable[i], camOrth));
				if (local > minDot)
				{
					minDot = local;
					minDotI = i;
				}
				if (local2 > maxDot)
				{
					maxDot = local2;
					maxDotI = i;
				}
			}

			glm::vec3 delta = moveable[minDotI] * glm::sign(glm::dot(moveable[minDotI], camUp)) * yDelta + 
				moveable[maxDotI] * glm::sign(glm::dot(moveable[maxDotI], camOrth)) * zDelta;
			//std::cout << moveable[maxDotI] << ":" << glm::sign(glm::dot(moveable[maxDotI], camUp)) << ":" << zDelta << std::endl;
			moveable.Translate(delta);
			/*auto fum = moveable.GetAABB();
			SlidingCollision slider{};
			for (auto& foobar : boxes.Search(fum))
			{
				if (foobar->box.Overlap(moveable, slider))
				{
					moveable.ApplyCollision(slider);
				}
			}*/
		}
	}



	mousePreviousX = x;
	mousePreviousY = y;
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
	windowWidth = width;
	windowHeight = height;
	aspectRatio = static_cast<float>(width) / height;
	
	cameraUniformBuffer.Generate(DynamicDraw, 2 * sizeof(glm::mat4));
	cameraUniformBuffer.SetBindingPoint(0);
	cameraUniformBuffer.BindUniform();

	glm::mat4 projection = glm::perspective(glm::radians(Fov * aspectRatio), aspectRatio, zNear, zFar);
	cameraUniformBuffer.BufferSubData(projection, sizeof(glm::mat4));

	depthed.GetColorBuffer<0>().CreateEmpty(windowWidth, windowHeight, InternalRGBA);
	depthed.GetColorBuffer<0>().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	depthed.GetColorBuffer<1>().CreateEmpty(windowWidth, windowHeight, InternalRGBA);
	depthed.GetColorBuffer<1>().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	depthed.GetDepth().CreateEmpty(windowWidth, windowHeight, InternalDepth);
	depthed.GetDepth().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	depthed.GetStencil().CreateEmpty(windowWidth, windowHeight, InternalStencil);
	// Doing NearestNearest is super messed up
	depthed.GetStencil().SetFilters(MinNearest, MagNearest, BorderClamp, BorderClamp);

	depthed.Assemble();

	toRemoveError.GetColor();

	scratchSpace.GetColorBuffer().CreateEmpty(windowWidth, windowHeight, InternalRGBA);
	scratchSpace.GetColorBuffer().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);
	scratchSpace.Assemble();

	screenSpaceBuffer.Generate(StaticRead, sizeof(glm::mat4));
	screenSpaceBuffer.SetBindingPoint(1);
	screenSpaceBuffer.BindUniform();
	screenSpaceBuffer.BufferSubData(glm::ortho<float>(0, static_cast<float>(windowWidth), static_cast<float>(windowHeight), 0));
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

void init();

constexpr std::size_t MapSize = 10;
static const std::array<const unsigned char, MapSize * MapSize> mapData =
{
	{
		0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 
		0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 
		0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
		0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF,
		0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
		0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x00,
		0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00,
		0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF,
		0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
	}
};

struct PathDummy
{
	unsigned char x, y;
	std::vector<std::weak_ptr<PathDummy>> dummies;
	std::vector<std::weak_ptr<PathDummy>> neighbors() const { return this->dummies; }
	constexpr PathDummy(unsigned char x = 0, unsigned char y = 0) : x(x), y(y) {}
	bool operator==(const PathDummy& other) const { return this->x == other.x && this->y == other.y; }
	float distance(const PathDummy& other) const { return static_cast<float>(glm::sqrt(glm::pow(this->x - other.x, 2) + glm::pow(this->y - other.y, 2))); }
	void AddNeighbor(const std::weak_ptr<PathDummy>& other) { this->dummies.push_back(other); }
};

namespace std
{
	template<> struct hash<PathDummy>
	{
		size_t operator()(const PathDummy& op) const
		{
			return static_cast<size_t>(op.x) << 16 | static_cast<size_t>(op.y) | static_cast<size_t>((op.y << 4) ^ op.y) << 8;
		}
	};
}
float heur(const PathDummy& dum1, const PathDummy& dum2) { return dum1.distance(dum2); }


int main(int argc, char** argv)
{
	/* Test for the accuracy of constexpr sqrt at runtime, 75% of the time it's dead on to double precision, 25% it's less than 1e13 off
	std::random_device r;
	std::default_random_engine randEngine(r());
	std::uniform_real_distribution distrib(1., 1000000.);
	double accumulator = 0, totalA = 0, totalB = 0;
	int fails = 0;
	int timesT = 2000000;
	double maxError = 0;
	glm::vec3 gone = Constexpr::normalize(glm::vec3(0, 1, 2));
	auto gone3 = Constexpr::length(glm::vec3(0, 1, 2));
	auto gone4 = Constexpr::dot(glm::vec3(0, 1, 2), glm::vec3(0, -4, 2));
	for (int i = 0; i < timesT; i++)
	{

		auto temp = distrib(randEngine);
		auto A = sqrt(temp);
		auto B = Constexpr::sqrt(temp);
		accumulator += abs(A - B);
		if (maxError < abs(A - B))
		{
			maxError = abs(A - B);
		}
		if (abs(A - B) != 0.f)
		{
			fails += 1;
		}
		//std::cout << sqrt(temp) << ":" << ConstexprSQRT(temp) << ":" << abs(sqrt(temp) - ConstexprSQRT(temp)) << std::endl;
	}
	std::cout << accumulator << " : " << accumulator / timesT << " : " << fails << " : " <<  float(fails) / timesT << std::endl;
	std::cout << "Max Error: " << maxError << std::endl;
	*/

	/* This is proof that dot of sum is equal to sum of dots, can be used to speed up OBB tests
	// This is wrong dumbass
	glm::vec3 a{}, b{}, c{}, d{}, ax{};
	float errored = 0.f;
	float maxError = 0.f;
	int iterations = 10000;
	for (int i = 0; i < iterations; i++)
	{
		a = glm::sphericalRand(1.);
		b = glm::normalize(glm::cross(glm::vec3(0, 1, 0), a));
		c = glm::normalize(glm::cross(a, b));

		a *= glm::linearRand(0.05f, 100.f);
		b *= glm::linearRand(0.05f, 100.f);
		c *= glm::linearRand(0.05f, 100.f);

		d = a + b + c;
		ax = glm::sphericalRand(1.);

		float dot = glm::abs(glm::dot(ax, d));
		//std::cout << dot << ":";
		float dotter = 0.;
		dotter += glm::abs(glm::dot(ax, a));
		dotter += glm::abs(glm::dot(ax, b));
		dotter += glm::abs(glm::dot(ax, c));
		//std::cout << glm::abs(dotter) << std::endl;
		if (glm::abs(dot - glm::abs(dotter)) > 1)
			std::cout << ax << ":" << dotter << ":" << dot << std::endl;
		dot = dot - glm::abs(dotter);
		//std::cout << abs(dot) << std::endl;
		maxError = glm::max(glm::abs(dot), maxError);
		errored += glm::abs(dot);
	}
	std::cout << "Max: " << maxError << std::endl;
	std::cout << "Average: " << errored / iterations << std::endl;
	std::cout << "Epsilon:" << EPSILON << std::endl;
	*/

	glm::vec3 a{}, b{}, c{}, d{}, ax{};
	float errored = 0.f;
	float maxError = 0.f;
	int iterations = 1;
	for (int i = 0; i < iterations; i++)
	{
		a = glm::normalize(glm::sphericalRand(1.f));
		b = glm::normalize(glm::sphericalRand(1.f));
		c = glm::normalize(glm::sphericalRand(1.f));

		/*
		a *= glm::linearRand(0.05f, 100.f);
		b *= glm::linearRand(0.05f, 100.f);
		c *= glm::linearRand(0.05f, 100.f);
		*/
		ax = glm::sphericalRand(1.);

		float original = glm::abs(glm::dot((glm::cross(a, b)), c));
		//std::cout << dot << ":";
		float dotter = glm::abs(glm::dot(a, (glm::cross(b, c))));
		
		original = original - glm::abs(dotter);
		//std::cout << abs(dot) << std::endl;
		maxError = glm::max(glm::abs(original), maxError);
		errored += glm::abs(original);
	}
	std::cout << "Max: " << maxError << std::endl;
	std::cout << "Average: " << errored / iterations << std::endl;
	std::cout << "Epsilon:" << EPSILON << std::endl;
	//return 0;

	int error = 0;
	debugFlags.fill(false);

	GLFWwindow* windowPointer = nullptr;
	if (!glfwInit())
	{
		LogF("Failed to initialized GLFW.n\n");
		return -1;
	}
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	glfwWindowHint(GLFW_OPENGL_API, GLFW_TRUE);
	glfwWindowHint(GLFW_STEREO, GLFW_FALSE);

	//glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API); // Look into it with 3.1 at some point
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


	glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_FALSE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

	windowPointer = glfwCreateWindow(windowWidth, windowHeight, "Wowie a window", nullptr, nullptr);
	if (!windowPointer)
	{
		Log("Failed to create window");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(windowPointer);

	int left, top, right, bottom;
	glfwGetWindowFrameSize(windowPointer, &left, &top, &right, &bottom);


	glfwSetWindowPos(windowPointer, 0, top);

	glewExperimental = GL_TRUE;
	// Glew
	if ((error = glewInit()) != GLEW_OK)
	{
		printf("Error code %i from glewInit()", error);
		return -1;
	}

	glfwSetKeyCallback(windowPointer, key_callback);

	glfwSetWindowFocusCallback(windowPointer, window_focus_callback);
	glfwSetWindowSizeCallback(windowPointer, window_size_callback);

	glfwSetMouseButtonCallback(windowPointer, mouseButtonFunc);
	glfwSetCursorPosCallback(windowPointer, mouseCursorFunc);

	init();
	window_size_callback(nullptr, windowWidth, windowHeight);

	while (!glfwWindowShouldClose(windowPointer))
	{
		idle();
		display();
		glfwSwapBuffers(windowPointer);
		glfwPollEvents();
	}
	// TODO: cleanup
	return 0;
}

void Dumber(std::size_t id) {}

void init()
{
	// OpenGL Feature Enabling
	EnableGLFeatures<DepthTesting | FaceCulling | DebugOutput>();
	DisableGLFeatures<MultiSampling>();
	glDepthFunc(GL_LEQUAL);

	glClearColor(0, 0, 0, 1);

	glFrontFace(GL_CCW);
	// OpenGL debuggin
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
	ground.CompileSimple("ground_");

	stencilTest.CompileSimple("stencil_");

	uniform.UniformBlockBinding("Camera", 0);
	dither.UniformBlockBinding("Camera", 0);
	flatLighting.UniformBlockBinding("Camera", 0);
	ground.UniformBlockBinding("Camera", 0);
	instancing.UniformBlockBinding("Camera", 0);
	sphereMesh.UniformBlockBinding("Camera", 0);

	stencilTest.UniformBlockBinding("Camera", 0);

	uiRect.UniformBlockBinding("ScreenSpace", 1);
	uiRectTexture.UniformBlockBinding("ScreenSpace", 1);
	fontShader.UniformBlockBinding("ScreenSpace", 1);

	// VAO SETUP
	fontVAO.ArrayFormat<UIVertex>(fontShader);
	
	instanceVAO.ArrayFormat<TextureVertex>(instancing, 0);
	instanceVAO.ArrayFormat<glm::mat4>(instancing, 1, 1);
	instanceVAO.ArrayFormat<TangentVertex>(instancing, 2);

	meshVAO.ArrayFormat<MeshVertex>(sphereMesh);

	normalVAO.ArrayFormat<NormalVertex>(flatLighting);

	//normalMapVAO.ArrayFormat<TangentVertex>(instancing, 2);

	plainVAO.ArrayFormat<Vertex>(uniform);

	texturedVAO.ArrayFormat<TextureVertex>(dither);

	// TEXTURE SETUP
	// TODO: texture loading base path thingy
	// These two textures from https://opengameart.org/content/stylized-mossy-stone-pbr-texture-set, do a better credit
	depthMap.Load("Textures/depth.png");
	depthMap.SetFilters(LinearLinear, MagLinear, MirroredRepeat, MirroredRepeat);
	depthMap.SetAnisotropy(16.f);

	ditherTexture.Load(dither16, InternalRed, FormatRed, DataUnsignedByte);
	ditherTexture.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	hatching.Load("Textures/hatching.png");
	hatching.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	normalMap.Load("Textures/normal.png");
	normalMap.SetFilters(LinearLinear, MagLinear, MirroredRepeat, MirroredRepeat);
	normalMap.SetAnisotropy(16.f);

	texture.Load("Textures/text.png");
	texture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);

	wallTexture.Load("Textures/flowed.png");
	wallTexture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);

	buttonB.CreateEmptyWithFilters(100, 100, InternalRGBA, glm::vec4(0, 1, 1, 1));
	buttonA.CreateEmptyWithFilters(100, 100, InternalRGBA, glm::vec4(1, 0.5, 1, 1));

	tessMap.Load(tesselationCode, InternalRed, FormatRed, DataUnsignedByte);
	tessMap.SetFilters(LinearLinear, MagLinear);

	/*
	mapper.Generate({ "Textures/skybox/right.jpg", "Textures/skybox/left.jpg", "Textures/skybox/top.jpg",
		"Textures/skybox/bottom.jpg", "Textures/skybox/front.jpg", "Textures/skybox/back.jpg" });
	*/

	stickBuffer.BufferData(stick, StaticDraw);

	std::array<TextureVertex, 4> verts{};
	//std::cout << sizeof(verts) << std::endl;
	//std::cout << sizeof(TextureVertex) << std::endl;
	//std::cout << sizeof(TextureVertex) * 4 << std::endl;

	for (int i = 0; i < 4; i++)
		verts[i].position = plane[i];
	verts[0].coordinates = glm::vec2(1, 1);
	verts[1].coordinates = glm::vec2(1, 0);
	verts[2].coordinates = glm::vec2(0, 1);
	verts[3].coordinates = glm::vec2(0, 0);
	
	std::array<TangentVertex, 4> tangents{};
	tangents.fill({ glm::vec3(1, 0, 0), glm::vec3(0, 0, 1) });

	normalMapBuffer.BufferData(tangents, StaticDraw);

	texturedPlane.BufferData(verts, StaticDraw);

	planeBO.BufferData(plane, StaticDraw);

	plainCube.BufferData(plainCubeVerts, StaticDraw);


	// RAY SETUP
	std::array<glm::vec3, 20> rays = {};
	rays.fill(glm::vec3(0));
	rayBuffer.BufferData(rays, StaticDraw);
	// CREATING OF THE PLANES

	for (int i = -5; i <= 5; i++)
	{
		if (abs(i) <= 1)
			continue;
		if (abs(i) == 3)
		{
			CombineVector(planes, GetPlaneSegment(glm::vec3(2 * i, 0, 0), PlusY));
			CombineVector(planes, GetPlaneSegment(glm::vec3(0, 0, 2 * i), PlusY));

			CombineVector(planes, GetPlaneSegment(glm::vec3(2 * i, 0, 2 * i), PlusY));
			CombineVector(planes, GetPlaneSegment(glm::vec3(-2 * i, 0, 2 * i), PlusY));
			for (int x = -2; x <= 2; x++)
			{
				if (x == 0)
					continue;
				CombineVector(planes, GetHallway(glm::vec3(2 * x, 0, 2 * i), false));
				CombineVector(planes, GetHallway(glm::vec3(2 * i, 0, 2 * x), true));
			}
			continue;
		}
		CombineVector(planes, GetHallway(glm::vec3(0, 0, 2 * i), true));
		CombineVector(planes, GetHallway(glm::vec3(2 * i, 0, 0), false));
	}
	for (int i = 0; i < 9; i++)
	{
		CombineVector(planes, GetPlaneSegment(glm::vec3(2 * (i % 3 - 1), 0, 2 * (static_cast<int>(i / 3) - 1)), PlusY));
	}
	// Diagonal Walls
	planes.push_back(Model(glm::vec3( 2, 1.f, -2), glm::vec3(0,  45,  90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2)))));
	planes.push_back(Model(glm::vec3( 2, 1.f,  2), glm::vec3(0, -45,  90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2)))));
	planes.push_back(Model(glm::vec3(-2, 1.f,  2), glm::vec3(0,  45, -90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2)))));
	planes.push_back(Model(glm::vec3(-2, 1.f, -2), glm::vec3(0, -45, -90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2)))));

	// The ramp
	planes.push_back(Model(glm::vec3(3.8f, .25f, 0), glm::vec3(0, 0.f, 15.0f), glm::vec3(1, 1, 1)));

	// The weird wall behind the player I think?
	planes.push_back(Model(glm::vec3(14, 1, 0), glm::vec3(0.f), glm::vec3(1.f, 20, 1.f)));

	std::vector<glm::mat4> awfulTemp{};
	awfulTemp.reserve(planes.size());
	//planes.push_back(Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f)));
	for (const auto& ref : planes)
	{
		OBB project(ref);
		//project.Scale(glm::vec3(1, .625f, 1));
		project.Scale(glm::vec3(1, 2e-6f, 1));
		boxes.Insert({project, false}, project.GetAABB());
		awfulTemp.push_back(ref.GetModelMatrix());
		//awfulTemp.push_back(ref.GetNormalMatrix());
	}

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
	albertBuffer.BufferData(textVert, StaticDraw);
	std::array<unsigned char, MapSize* MapSize> copied{};
	copied.fill(0x00);
	std::vector<std::shared_ptr<PathDummy>> sleepers;
	// Make Nodes
	for (std::size_t y = 0; y < MapSize; y++)
	{
		for (std::size_t x = 0; x < MapSize; x++)
		{
			copied[x + y * MapSize] = (mapData[x + y * MapSize] == 0xFF) ? 0x80 : 0x00;
			auto FAM = std::make_shared<PathDummy>();
			FAM->x = static_cast<unsigned char>(x);
			FAM->y = static_cast<unsigned char>(y);
			sleepers.push_back(FAM);
		}
	}
	// Fill Neighbors
	for (std::size_t y = 0; y < MapSize; y++)
	{
		for (std::size_t x = 0; x < MapSize; x++)
		{
			if (mapData[x + y * MapSize] == 0xFF)
			{
				auto& current = sleepers[x + y * MapSize];
				if (x > 0 && mapData[(x - 1) + y * MapSize] == 0xFF)
				{
					current->AddNeighbor(sleepers[(x - 1) + y * MapSize]);
					//sleepers[(x - 1) + y * MapSize]->AddNeighbor(current);
				}
				if (x < (MapSize - 1) && mapData[(x + 1) + y * MapSize] == 0xFF)
				{
					current->AddNeighbor(sleepers[(x + 1) + y * MapSize]);
					//sleepers[(x + 1) + y * MapSize]->AddNeighbor(current);
				}
				if (y > 0 && mapData[x + (y - 1) * MapSize] == 0xFF)
				{
					current->AddNeighbor(sleepers[x + (y - 1) * MapSize]);
					//sleepers[x + (y - 1) * MapSize]->AddNeighbor(current);
				}
				if (y < (MapSize - 1) && mapData[x + (y + 1) * MapSize] == 0xFF)
				{
					current->AddNeighbor(sleepers[x + (y + 1) * MapSize]);
					//sleepers[x + (y + 1) * MapSize]->AddNeighbor(current);
				}
			}
		}
	}
	auto& te2 = sleepers[0];
	auto& te3 = sleepers.back();
	auto losers = AStarSearch<PathDummy>(te2, te3, heur);
	//auto losers = AStarSearch<PathDummy>(te2, te3, [](const PathDummy& a, const PathDummy& b) {return 0.f; });
	for (auto& flam : losers.second)
	{
		copied[flam->x + MapSize * flam->y] = 0xC0;
	}
	for (auto& flam : losers.first)
	{
		copied[flam->x + MapSize * flam->y] = 0xFF;
		//std::cout << static_cast<unsigned int>(flam->x) << ":" << static_cast<unsigned int>(flam->y) << std::endl;
	}
	buttonB.Load(copied, InternalRed, FormatRed, DataUnsignedByte);


	// FRAMEBUFFER SETUP
	// TODO: Renderbuffer for buffers that don't need to be directly read

	// This was moved to the window-resizing place, TODO: see if that's the bottleneck on startup


	Sphere::GenerateMesh(sphereBuffer, sphereIndicies, 30, 30);
	Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.1f, 10.f, 30, 30);
	Capsule::GenerateMesh(movingCapsule, movingCapsuleIndex, 0.25f, 0.5f, 30, 30);

	catapult.SetCenter(glm::vec3(0, 0.5f, 0));
	catapult.SetRadius(0.25f);
	catapult.SetLength(0.5f);

	catapultModel.translation = glm::vec3(0, 0.5f, 0);
	catapultBox.ReCenter(glm::vec3(0, 0.5, 0));
	catapultBox.ReScale(glm::vec3(0.25f, 0.5f, 0.25f));


	Font::SetFontDirectory("Fonts");

	// Awkward syntax :(
	ASCIIFont::LoadFont(fonter, "CommitMono-400-Regular.ttf", 50.f, 2, 2);

	stickIndicies.BufferData(stickDex, StaticDraw);

	cubeOutlineIndex.BufferData(cubeOutline, StaticDraw);

	smartBox.ReScale(glm::vec3(0.5f));
	smartBox.Scale(1.1f);

	smartReset();
	smartBox.ReCenter(glm::vec3(1.2f, 0.5f, 0));
	//smartBox.ReCenter(glm::vec3(12.2f, 1.6f, 0));
	smartBox.ReOrient(glm::vec3(0, 0, 0));
	dumbBox.ReCenter(glm::vec3(0, 1.f, -2));
	dumbBox.Scale(glm::vec3(1.f));
	dumbBox.Rotate(glm::vec3(0, -90, 0));

	moveable.ReCenter(glm::vec3(0, .25, 0));
	moveable.Scale(0.25f);

	loom.ReCenter(glm::vec3(0, 5, 0));
	loom.ReOrient(glm::vec3(0.f, 0, 90.f));
	//boxes.Insert({ dumbBox, false }, dumbBox.GetAABB());
	//Log("Doing it");
	//windowResize(1000, 1000);
	Button buttonMan({ 0, 0, 20, 20 }, Dumber);
	fonter.Render(buttonA, glm::vec2(), "Soft");
	//fonter.Render(buttonB, glm::vec2(), "Not");
	
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glClearColor(0.f, 0.f, 0.f, 0.f);
}