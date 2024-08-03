#include <algorithm>
#include <chrono>
#include <glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/ulp.hpp>
#include <glm/gtx/euler_angles.hpp>
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
#include "Decal.h"
#include "DynamicTree.h"
#include "Font.h"
#include "Framebuffer.h"
#include "glmHelp.h"
#include "glUtil.h"
#include "Lines.h"
#include "log.h"
#include "Input.h"
#include "Model.h"
#include "OBJReader.h"
#include "OrientedBoundingBox.h"
#include "Pathfinding.h"
#include "PathFollower.h"
#include "PathNode.h"
#include "Plane.h"
#include "QuickTimer.h"
#include "Shader.h"
#include "ScreenRect.h"
#include "Sphere.h"
#include "StaticOctTree.h"
#include "stbWrangler.h"
#include "Texture2D.h"
#include "Triangle.h"
#include "UniformBuffer.h"
#include "util.h"
#include "Vertex.h"
#include "VertexArray.h"
#include "UserInterface.h"
#include "TextureUtil.h"
#include "Window.h"

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
// TODO: https://github.com/zeux/meshoptimizer once you use meshes
// TODO: imGUI
// TODO: Delaunay Trianglulation
// TODO: EASTL


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
/*
Assume we have the reduced form with only 24 verticies in the order laid out above, we get:
{
	0, 1, 2,  1, 3, 2, // -X Face
	4, 5, 6,  5, 7, 6, // -Y Face

}

*/

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
ArrayBuffer albertBuffer, textBuffer, capsuleBuffer, instanceBuffer, plainCube, planeBO, rayBuffer, sphereBuffer, stickBuffer, texturedPlane;
ArrayBuffer cubeMesh, movingCapsule, normalMapBuffer;
ArrayBuffer pathNodePositions, pathNodeLines;
ArrayBuffer decals;

ElementArray capsuleIndex, cubeOutlineIndex, movingCapsuleIndex, solidCubeIndex, sphereIndicies, stickIndicies;

UniformBuffer cameraUniformBuffer, pointUniformBuffer, screenSpaceBuffer;

// Framebuffer
Framebuffer<2, DepthAndStencil> depthed;
ColorFrameBuffer scratchSpace;

// Shaders
Shader dither, expand, finalResult, flatLighting, fontShader, frameShader, ground, instancing, uiRect, uiRectTexture, uniform, sphereMesh, widget;
Shader triColor, decalShader;
Shader pathNodeView, stencilTest;
Shader nineSlicer;
Shader skinner;
Shader billboardShader;
Shader voronoi;

// Textures
Texture2D depthMap, ditherTexture, hatching, normalMap, tessMap, texture, wallTexture;
Texture2D buttonA, buttonB, nineSlice;
CubeMap mapper;

// Vertex Array Objects
VAO decalVAO, fontVAO, instanceVAO, pathNodeVAO, meshVAO, normalVAO, normalMapVAO, plainVAO, texturedVAO;
VAO nineSliced;
VAO instanceVAO2;

// Not explicitly tied to OpenGL Globals

OBB dumbBox; // rip smartbox
std::vector<Model> instancedModels;
std::vector<MaxHeapValue<OBB>> instancedDrawOrder;
glm::vec3 lastCameraPos;

StaticOctTree<OBB> staticBoxes(glm::vec3(20));

static unsigned int frameCounter = 0;

glm::vec3 movingSphere(0, 3.5f, 6.5f);
int lineWidth = 3;

float aspectRatio = 1.f;
static const float Fov = 70.f;

#define TIGHT_BOXES 1
#define WIDE_BOXES 2
#define DEBUG_PATH 3
#define DYNAMIC_TREE 4
// One for each number key
std::array<bool, '9' - '0' + 1> debugFlags{};

// Input Shenanigans
#define ArrowKeyUp    0
#define ArrowKeyDown  1
#define ArrowKeyRight 2
#define ArrowKeyLeft  3

std::array<bool, UCHAR_MAX> keyState{}, keyStateBackup{};



ColorFrameBuffer playerTextEntry;
std::stringstream letters("abc");
bool reRenderText = true;

constexpr float ANGLE_DELTA = 4;


// Camera
glm::vec3 cameraPosition(0, 1.5f, 0);
glm::vec3 cameraRotation(0, 0, 0);

float zNear = 0.1f, zFar = 100.f;

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

void GetPlaneSegment(const glm::vec3& base, GeometryThing flags, std::vector<Model>& results)
{
	if (flags & PlusX)  results.emplace_back(base + glm::vec3(-1, 1,  0), glm::vec3(  0, 0, -90.f));
	if (flags & MinusX) results.emplace_back(base + glm::vec3( 1, 1,  0), glm::vec3(  0, 0,  90.f));
	if (flags & PlusZ)  results.emplace_back(base + glm::vec3( 0, 1, -1), glm::vec3( 90, 0,     0));
	if (flags & MinusZ) results.emplace_back(base + glm::vec3( 0, 1,  1), glm::vec3(-90, 0,     0));
	if (flags & PlusY)  results.emplace_back(base);
	if (flags & MinusY) results.emplace_back(base + glm::vec3( 0, 2,  0), glm::vec3(180, 0,     0));
}

void GetHallway(const glm::vec3& base, std::vector<Model>& results, bool openZ = true)
{
	GetPlaneSegment(base, (openZ) ? HallwayZ : HallwayX, results);
}

bool buttonToggle = false;
ScreenRect buttonRect{ 540, 200, 100, 100 }, userPortion(0, 800, 1000, 200);
Button help(buttonRect, [](std::size_t i) {std::cout << frameCounter << std::endl; });


Capsule catapult;
Model catapultModel;
OBB catapultBox;

// TODO: Line Shader with width, all the math being on gpu (given the endpoints and the width then do the orthogonal to the screen kinda thing)
// TODO: Move cube stuff into a shader or something I don't know

OBB pointingCapsule;

OBB moveable;

int tessAmount = 5;

bool featureToggle = false;
std::chrono::nanoseconds idleTime, displayTime, renderDelay;

constexpr float BulletRadius = 0.05f;
struct Bullet
{
	glm::vec3 position, direction;
};

PathFollower followed{glm::vec3(0, 0.5f, 0) };


DynamicOctTree<PathFollower> followers{AABB(glm::vec3(-105), glm::vec3(100))};

std::vector<Bullet> bullets;

std::vector<TextureVertex> decalVertex;

std::array<ScreenRect, 9> ui_tester;
ArrayBuffer ui_tester_buffer;

/*
New shading outputs
-A stencil lights factor(mix(current, light, factor))
-

*/

std::array<glm::mat4, 2> skinMats;
ArrayBuffer skinBuf;
ArrayBuffer skinVertex;
ElementArray skinArg;

ArrayBuffer billboardBuffer;

std::vector<AABB> dynamicTreeBoxes;
using namespace Input;

void display()
{
	auto displayStartTime = std::chrono::high_resolution_clock::now();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//depthed.Bind();
	glViewport(0, 0, Window::Width, Window::Height);
	glClearColor(0, 0, 0, 1);

	EnableGLFeatures<DepthTesting | FaceCulling>();
	EnableDepthBufferWrite();
	ClearFramebuffer<ColorBuffer | DepthBuffer | StencilBuffer>();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	DisableGLFeatures<StencilTesting>();
	
	// Camera matrix
	glm::vec3 cameraRadians = glm::radians(cameraRotation);

	// Adding pi/2 is necessary because the default camera is facing -z
	glm::mat4 view = glm::translate(glm::eulerAngleXYZ(cameraRadians.x, cameraRadians.y + glm::half_pi<float>(), cameraRadians.z), 
									-cameraPosition);
	cameraUniformBuffer.BufferSubData(view, 0);
	
	DisableGLFeatures<Blending>();
	instancing.SetActiveShader();
	instancing.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	instancing.SetVec3("lightPos", followed.GetPosition());
	instancing.SetVec3("viewPos", cameraPosition);
	instancing.SetTextureUnit("textureIn", wallTexture, 0);
	instancing.SetTextureUnit("ditherMap", ditherTexture, 1);
	instancing.SetTextureUnit("normalMapIn", normalMap, 2);
	instancing.SetTextureUnit("depthMapIn", depthMap, 3);
	instancing.SetInt("newToggle", featureToggle);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	// Maybe move this elsewhere
	if (frameCounter % 500 == 0 && lastCameraPos != cameraPosition)
	{
		lastCameraPos = cameraPosition;
		QUICKTIMER("DepthSorting");
		for (auto& p : instancedDrawOrder)
		{
			p.value = p.element.SignedDistance(cameraPosition);
		}
		std::sort(instancedDrawOrder.begin(), instancedDrawOrder.end());
		std::vector<glm::mat4> combo;
		combo.reserve(instancedModels.size());
		for (auto& p : instancedDrawOrder)
		{
			glm::mat4 fumo = p.element.GetModelMatrix();
			fumo[1] = glm::vec4(p.element[1], 0);
			combo.push_back(fumo);
		}
		instanceBuffer.BufferData(combo, StaticDraw);
	}

	instanceVAO.BindArrayBuffer(texturedPlane, 0);
	instanceVAO.BindArrayBuffer(instanceBuffer, 1);
	instanceVAO.BindArrayBuffer(normalMapBuffer, 2);
	instancing.DrawArrayInstanced<DrawType::TriangleStrip>(texturedPlane, instanceBuffer);

	if (debugFlags[DEBUG_PATH])
	{
		EnableGLFeatures<Blending>();
		DisableDepthBufferWrite();
		pathNodeView.SetActiveShader();
		pathNodeVAO.BindArrayObject();
		pathNodeVAO.BindArrayBuffer(plainCube, 0);
		pathNodeVAO.BindArrayBuffer(pathNodePositions, 1);
		pathNodeView.SetFloat("Scale", (glm::cos(frameCounter / 200.f) * 0.05f) + 0.3f);
		pathNodeView.SetVec4("Color", glm::vec4(0, 0, 1, 0.75f));
		
		pathNodeView.DrawElementsInstanced<DrawType::Triangle>(solidCubeIndex, pathNodePositions);

		uniform.SetActiveShader();
		uniform.SetMat4("Model", glm::mat4(1.f));
		plainVAO.BindArrayBuffer(pathNodeLines);
		glLineWidth(10.f);
		uniform.DrawArray<DrawType::Lines>(pathNodeLines);

		EnableDepthBufferWrite();
	}
	// Visualize the pathfinder guy
	{
		EnableGLFeatures<Blending>();
		DisableDepthBufferWrite();
		pathNodeView.SetActiveShader();
		pathNodeVAO.BindArrayObject();
		pathNodeVAO.BindArrayBuffer(plainCube, 0);
		pathNodeVAO.BindArrayBuffer(PathFollower::latestPathBuffer, 1);
		pathNodeView.SetFloat("Scale", (glm::cos(frameCounter / 200.f) * 0.05f) + 0.3f);
		pathNodeView.SetVec4("Color", glm::vec4(0, 0, 1, 0.75f));

		pathNodeView.DrawElementsInstanced<DrawType::Triangle>(solidCubeIndex, PathFollower::latestPathBuffer);

		uniform.SetActiveShader();
		uniform.SetMat4("Model", glm::mat4(1.f));
		plainVAO.BindArrayBuffer(PathFollower::latestPathBuffer);
		glLineWidth(10.f);
		uniform.DrawArray<DrawType::LineStrip>(PathFollower::latestPathBuffer);
		EnableDepthBufferWrite();
	}


	/* STICK FIGURE GUY */
	uniform.SetActiveShader();
	plainVAO.BindArrayBuffer(stickBuffer);

	glm::vec3 colors = glm::vec3(1, 0, 0);
	Model m22(glm::vec3(10, 0, 0));
	uniform.SetMat4("Model", m22.GetModelMatrix());
	uniform.SetVec3("color", colors);
	uniform.DrawElements<DrawType::LineStrip>(stickIndicies);

	DisableGLFeatures<FaceCulling>();
	skinner.SetActiveShader();
	instanceVAO2.BindArrayObject();
	instanceVAO2.BindArrayBuffer(skinVertex, 0);
	skinner.SetMat4s("mats", std::span{skinMats});
	skinner.SetTextureUnit("textureIn", wallTexture, 0);
	skinArg.BindBuffer();
	//skinner.DrawElements<DrawType::Triangle>(skinArg);
	

	EnableGLFeatures<FaceCulling>();
	billboardShader.SetActiveShader();
	texturedVAO.BindArrayBuffer(billboardBuffer);
	billboardShader.SetTextureUnit("sampler", texture, 0);
	auto yCameraMatrix = glm::eulerAngleY(-cameraRadians.y);
	billboardShader.SetMat4("orient", yCameraMatrix);
	glm::vec3 radians = -glm::radians(cameraRotation);
	glm::mat4 cameraOrientation = glm::eulerAngleXYZ(radians.z, radians.y, radians.x);

	for (auto& following : followers)
	{
		glm::vec3 billboardPosition = following.first.GetPosition();
		glm::vec3 billboardDelta = cameraPosition - billboardPosition;
		glm::vec3 up = glm::vec3(0, 1, 0);
		glm::vec3 right = glm::normalize(glm::cross(up, billboardDelta));
		glm::vec3 forward = glm::normalize(glm::cross(right, up));
		glm::mat4 tempMatrix{ glm::mat3(forward, up, right)};
		tempMatrix = glm::transpose(tempMatrix);
		tempMatrix[3] = glm::vec4(billboardPosition, 1);
		tempMatrix[2] *= 0.5f;
		billboardShader.SetMat4("orient", tempMatrix);
		//billboardShader.DrawArray<DrawType::TriangleStrip>(billboardBuffer);
	}
	/*
	DisableGLFeatures<FaceCulling>();
	ground.SetActiveShader();
	glPatchParameteri(GL_PATCH_VERTICES, 4);
	texturedVAO.BindArrayBuffer(texturedPlane);
	ground.SetTextureUnit("heightMap", tessMap, 0);
	float big = 30;
	m22.scale = glm::vec3(big);
	ground.SetMat4("Model", m22.GetModelMatrix());
	ground.SetInt("redLine", 0);
	ground.SetInt("amount", tessAmount);
	for (int i = -5; i <= 5; i++)
	{
		for (int j = -5; j <= 5; j++)
		{
			m22.translation = glm::vec3(i, 0, j) * 2.f * big;
			m22.translation.y = 3;
			ground.SetMat4("Model", m22.GetModelMatrix());
			//ground.SetInt("redLine", 0);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			//ground.DrawArray<DrawType::Patches>(4);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			ground.SetInt("redLine", 1);
			ground.DrawArray<DrawType::Patches>(4);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
	}
	EnableGLFeatures<FaceCulling>();
	*/

	decalShader.SetActiveShader();
	decalVAO.BindArrayBuffer(decals);
	decalShader.SetTextureUnit("textureIn", texture, 0);
	decalShader.DrawArray<DrawType::Triangle>(decals);


	if (debugFlags[DYNAMIC_TREE])
	{
		uniform.SetActiveShader();
		glm::vec3 blue(0, 0, 1);
		plainVAO.BindArrayBuffer(plainCube);
		uniform.SetVec3("color", glm::vec3(1, 0.65, 0));
		for (auto& box : dynamicTreeBoxes)
		{
			auto d = box.GetModel();
			d.scale *= 0.99f;
			uniform.SetMat4("Model", d.GetModelMatrix());
			uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
		}
	}

	// Debugging staticBoxes
	if (debugFlags[TIGHT_BOXES] || debugFlags[WIDE_BOXES])
	{
		uniform.SetActiveShader();
		glm::vec3 blue(0, 0, 1);
		plainVAO.BindArrayBuffer(plainCube);

		OBB placeholder(AABB(glm::vec3(0), glm::vec3(1)));
		placeholder.Translate(glm::vec3(2, 0.1, 0));
		placeholder.Rotate(glm::radians(glm::vec3(frameCounter * -2.f, frameCounter * 4.f, frameCounter)));
		uniform.SetMat4("Model", placeholder.GetModelMatrix());
		uniform.SetVec3("color", blue);

		float wid = 10;
		if (debugFlags[TIGHT_BOXES]) uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
		uniform.SetMat4("Model", placeholder.GetAABB().GetModel().GetModelMatrix());
		uniform.SetVec3("color", glm::vec3(0.5f, 0.5f, 0.5f));

		if (debugFlags[WIDE_BOXES]) uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
		for (const auto& box: staticBoxes)
		{
			if (debugFlags[TIGHT_BOXES])
			{
				uniform.SetMat4("Model", box.GetModelMatrix());
				//uniform.DrawElementsMemory<Triangle>(cubeIndicies);
				uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
				uniform.DrawArray<DrawType::Points>(8);
			}
			if (debugFlags[WIDE_BOXES])
			{
				uniform.SetMat4("Model", box.GetAABB().GetModel().GetModelMatrix());
				uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
				//uniform.DrawArray<Points>(8);
			}
		}
	}

	// Cubert
	uniform.SetActiveShader();
	plainVAO.BindArrayBuffer(plainCube);
	uniform.SetMat4("Model", dumbBox.GetModelMatrix());
	
	glDepthMask(GL_TRUE);
	uniform.SetVec3("color", glm::vec3(1, 1, 1));
	moveable.ReScale(glm::vec3(0, 1, 1));

	//glDepthMask(GL_FALSE)
	// Albert
	
	//glPatchParameteri(GL_PATCH_VERTICES, 3);
	texturedVAO.BindArrayBuffer(albertBuffer);
	dither.SetActiveShader();
	dither.SetTextureUnit("ditherMap", wallTexture, 1);
	dither.SetTextureUnit("textureIn", texture, 0);
	dither.SetMat4("Model", dumbBox.GetModelMatrix());
	dither.SetVec3("color", glm::vec3(0, 1, 0));
	dither.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	dither.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	dither.SetVec3("viewPos", cameraPosition);
	//dither.DrawArray<DrawType::Triangle>(36);
	dither.DrawArray<DrawType::Patches>(albertBuffer);


	plainVAO.BindArrayBuffer(plainCube);
	uniform.SetActiveShader();
	uniform.SetMat4("Model", dumbBox.GetModelMatrix());
	uniform.SetMat4("Model", catapult.GetAABB().GetModel().GetModelMatrix());
	//uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);

	// Drawing of the rays
	//DisableGLFeatures<DepthTesting>();
	plainVAO.BindArrayBuffer(rayBuffer);
	Model bland;
	uniform.SetMat4("Model", bland.GetModelMatrix());
	glLineWidth(15.f);
	uniform.DrawArray<DrawType::Lines>(rayBuffer);
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
	sphereMesh.DrawElements<DrawType::Triangle>(sphereIndicies);
	for (auto& bullet : bullets)
	{
		Model localModel;
		localModel.translation = bullet.position;
		localModel.scale = glm::vec3(BulletRadius);
		sphereMesh.SetMat4("modelMat", localModel.GetModelMatrix());
		sphereMesh.SetMat4("normalMat", localModel.GetNormalMatrix());
		sphereMesh.SetTextureUnit("textureIn", texture, 0);
		//mapper.BindTexture(0);
		//sphereMesh.SetTextureUnit("textureIn", 0);
		sphereMesh.DrawElements<DrawType::Triangle>(sphereIndicies);
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
	stencilTest.DrawElements<DrawType::Triangle>(sphereIndicies);

	plainVAO.BindArrayBuffer(plainCube);
	stencilTest.SetMat4("Model", lightModel.GetModelMatrix());
	stencilTest.DrawElementsMemory<DrawType::Triangle>(cubeIndicies);

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
	uiRect.SetVec4("rectangle", glm::vec4(0, 0, Window::Width, Window::Height));
	//uiRect.DrawArray(TriangleStrip, 4);
	//uiRect.DrawArray(TriangleStrip, 4);

	DisableGLFeatures<StencilTesting>();
	//EnableGLFeatures<DepthTesting>();
	glDepthMask(GL_TRUE);

	flatLighting.SetActiveShader();
	meshVAO.BindArrayBuffer(capsuleBuffer);
	flatLighting.SetVec3("lightColor", glm::vec3(1.f, 0.f, 0.f));
	flatLighting.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	flatLighting.SetVec3("viewPos", cameraPosition);
	flatLighting.SetMat4("modelMat", pointingCapsule.GetNormalMatrix());
	flatLighting.SetMat4("normalMat", pointingCapsule.GetNormalMatrix());
	//flatLighting.SetVec3("shapeColor", glm::vec3(0.8f, 0.34f, 0.6f));
	flatLighting.SetVec3("shapeColor", glm::vec3(0.f, 0.f, 0.8f));
	flatLighting.DrawElements<DrawType::Triangle>(capsuleIndex);


	meshVAO.BindArrayBuffer(movingCapsule);
	flatLighting.SetMat4("modelMat", followed.GetNormalMatrix());
	flatLighting.SetMat4("normalMat", followed.GetNormalMatrix());
	flatLighting.DrawElements<DrawType::Triangle>(movingCapsuleIndex);
	// Calling with triangle_strip is fucky
	/*
	flatLighting.DrawElements(Triangle, sphereIndicies);
	sphereModel.translation = movingSphere;
	flatLighting.SetMat4("modelMat", sphereModel.GetModelMatrix());
	flatLighting.SetMat4("normMat", sphereModel.GetNormalMatrix());
	flatLighting.DrawElements(Triangle, sphereIndicies);
	*/

	DisableGLFeatures<DepthTesting>();
	EnableGLFeatures<Blending>();
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	uiRect.SetActiveShader();
	uiRect.SetVec4("color", glm::vec4(0, 0.5, 0.75, 0.25));
	uiRect.SetVec4("rectangle", glm::vec4(0, 0, 200, 100));
	uiRect.DrawArray<DrawType::TriangleStrip>(4);
	
	uiRect.SetVec4("rectangle", glm::vec4(Window::Width - 200, 0, 200, 100));
	uiRect.DrawArray<DrawType::TriangleStrip>(4);
	
	uiRect.SetVec4("rectangle", glm::vec4(0, Window::Height - 100, 200, 100));
	uiRect.DrawArray<DrawType::TriangleStrip>(4);
	
	uiRect.SetVec4("rectangle", glm::vec4(Window::Width - 200, Window::Height - 100, 200, 100));
	uiRect.DrawArray<DrawType::TriangleStrip>(4);

	uiRect.SetVec4("rectangle", userPortion);
	uiRect.SetVec4("color", glm::vec4(0.25, 0.25, 0.25, 0.85));
	//uiRect.DrawArray<DrawType::TriangleStrip>(4);

	nineSlicer.SetActiveShader();
	nineSlicer.SetTextureUnit("image", nineSlice);
	nineSliced.BindArrayBuffer(ui_tester_buffer);
	nineSlicer.DrawArrayInstanced<DrawType::TriangleStrip>(4, 9);


	uiRectTexture.SetActiveShader();
	auto& colored = playerTextEntry.GetColor();
	uiRectTexture.SetTextureUnit("image", colored, 0);
	uiRectTexture.SetVec4("rectangle", glm::vec4((Window::Width - colored.GetWidth()) / 2, (Window::Height - colored.GetHeight()) / 2,
		colored.GetWidth(), colored.GetHeight()));
	uiRect.DrawArray<DrawType::TriangleStrip>(4);

	uiRectTexture.SetTextureUnit("image", (buttonToggle) ? buttonA : buttonB, 0);
	uiRectTexture.SetVec4("rectangle", buttonRect);
	uiRect.DrawArray<DrawType::TriangleStrip>(4);

	uiRectTexture.SetTextureUnit("image", help.GetTexture(), 0);
	uiRectTexture.SetVec4("rectangle", help.GetRect());
	uiRect.DrawArray<DrawType::TriangleStrip>(4);

	uiRectTexture.SetTextureUnit("image", normalMap);
	uiRectTexture.SetVec4("rectangle", { 0, 0, normalMap.GetSize()});
	//uiRect.DrawArray<DrawType::TriangleStrip>(4);

	DisableGLFeatures<FaceCulling>();
	DisableGLFeatures<Blending>();
	voronoi.SetActiveShader();
	voronoi.SetInt("mode", 0);
	//voronoi.DrawArray<DrawType::TriangleStrip>(4);

	EnableGLFeatures<Blending>();
	// Debug Info Display
	fontShader.SetActiveShader();
	fontVAO.BindArrayBuffer(textBuffer);
	fontShader.SetTextureUnit("fontTexture", fonter.GetTexture(), 0);
	fontShader.DrawArray<DrawType::Triangle>(textBuffer);

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
	frameShader.DrawArray<TriangleStrip>(4);
	*/
	/*
	BindDefaultFrameBuffer();
	glClearColor(1, 0.5, 0.25, 1);
	ClearFramebuffer<DepthBuffer | StencilBuffer | ColorBuffer>();

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
	expand.SetInt("flag", false);
	expand.DrawArray<DrawType::TriangleStrip>(4);
	*/
	glStencilMask(0xFF);

	glLineWidth(1.f);
	widget.SetActiveShader();
	widget.DrawArray<DrawType::Lines>(6);
	
	EnableGLFeatures<DepthTesting | StencilTesting | FaceCulling>();

	auto end = std::chrono::high_resolution_clock::now();
	displayTime = end - displayStartTime;
	displayStartTime = end;
	glFinish();
	end = std::chrono::high_resolution_clock::now();
	renderDelay = end - displayStartTime;
}

static const glm::vec3 GravityAxis{ 0.f, -1.f, 0.f };
static const glm::vec3 GravityUp{ 0.f, 1.f, 0.f };

OBB* capsuleHit;
glm::vec3 capsuleNormal, capsuleAcceleration, capsuleVelocity;
int shift = 2;

// TODO: Mech suit has an interior for the pilot that articulates seperately from the main body, within the outer limits of the frame
// Like it's a bit pliable
void idle()
{
	static auto lastIdleStart = std::chrono::high_resolution_clock::now();
	static std::deque<float> frames;
	static std::deque<long long> displayTimes, idleTimes, renderTimes;
	static unsigned long long displaySimple = 0, idleSimple = 0;

	frameCounter++;
	const auto idleStart = std::chrono::high_resolution_clock::now();
	const auto delta = idleStart - lastIdleStart;

	const float timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(delta).count();
	
	auto idleDelta = idleTime.count() / 1000;
	auto displayDelta = displayTime.count() / 1000;
	auto renderDelta = renderDelay.count() / 1000;
	frames.push_back(1.f / timeDelta);
	displayTimes.push_back(displayDelta);
	idleTimes.push_back(idleDelta);
	renderTimes.push_back(renderDelta);
	if (frames.size() > 300)
	{
		frames.pop_front();
		displayTimes.pop_front();
		idleTimes.pop_front();
		renderTimes.pop_front();
	}
	float averageFps = 0.f;
	long long averageIdle = 0, averageDisplay = 0, averageRender = 0;
	for (std::size_t i = 0; i < frames.size(); i++)
	{
		averageFps += frames[i] / frames.size();
		averageDisplay += displayTimes[i] / displayTimes.size();
		averageIdle += idleTimes[i] / idleTimes.size();
		averageRender += renderTimes[i] / renderTimes.size();
	}
	// This will be used for "release" mode as it's faster but noisier
	if (!idleSimple) idleSimple = idleDelta;
	if (!displaySimple) displaySimple = displayDelta;
	idleSimple =  (idleSimple / 2) + (idleDelta / 2);
	displaySimple =  (displaySimple / 2) + (displayDelta / 2);
	// End of Rolling buffer

	float speed = 4 * timeDelta;
	float turnSpeed = 100 * timeDelta;

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
		OBB playerBox(playerBounds);
		playerBounds.Center(cameraPosition);

		OBB playerObb(playerBounds);
		//playerObb.Rotate(glm::eulerAngleY(glm::radians(-cameraRotation.y)));
		playerObb.Rotate(glm::vec3(0, -cameraRotation.y, 0));

		playerBox.Translate(glm::vec3(2, 0, 0));
		playerBox.Rotate(glm::vec3(0, frameCounter * 4.f, 0));
		for (auto& wall : staticBoxes)
		{
			if (wall.Overlap(playerObb))
			{
				playerObb.OverlapAndSlide(wall);
				//offset = previous;
				//break;
			}
		}
		if (playerBox.Overlap(playerObb))
		{
			//offset = previous;
		}
		cameraPosition = playerObb.Center();
		//Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f))
	}

	// Physics attempt
	const float CapsuleAcceleration = 0.06f;
	const float CapsuleMass = 1;
	const float staticFrictionCoeff = 1.0f;
	const float slidingFrictionCoeff = 0.57f;
	const float GravityRate = 0.25f;
	const float BoxGravityMagnitude = CapsuleMass * GravityRate;
	const glm::vec3 capsuleGravity = GravityAxis * BoxGravityMagnitude;

	// CAPSULE STUFF
	float mult = float(keyState[ArrowKeyUp] ^ keyState[ArrowKeyDown]) * ((keyState[ArrowKeyDown]) ? -1.f : 1.f);
	float capsuleDot = -INFINITY;
	glm::vec3 capsuleForces{};
	// Transformations need to be addressed
	if (!capsuleHit)
	{
		capsuleForces += capsuleGravity;
	}
	capsuleForces += catapultBox.Forward() * mult * CapsuleAcceleration;
	capsuleHit = nullptr;
	capsuleNormal = glm::vec3(0);
	capsuleAcceleration = capsuleForces / CapsuleMass * timeDelta;
	capsuleVelocity += capsuleAcceleration;
	//std::cout << catapult.GetCenter() << std::endl;
	if (glm::length(capsuleVelocity) > 2.f)
		capsuleVelocity = glm::normalize(capsuleVelocity) * 2.f;
	catapult.Translate(capsuleVelocity);
	capsuleVelocity *= 0.99f; // Maybe "real" friction?
	for (auto& temps : staticBoxes.Search(catapult.GetAABB()))
	{
		Collision c;
		if (temps->Overlap(catapult, c))
		{
			catapult.Translate(c.normal * c.depth);
			float dot = glm::dot(GravityUp, c.normal);
			if (dot > 0 && dot > capsuleDot)
			{
				capsuleHit = &*temps;
				capsuleDot = dot;
			}
		}
	}
	catapultBox.ReCenter(catapult.GetCenter());


	Sphere spherePlaceholder(0.5f, movingSphere);
	for (auto& letsgo : staticBoxes.Search(spherePlaceholder.GetAABB()))
	{
		Collision c;
		if (letsgo->Overlap(spherePlaceholder, c))
		{
			spherePlaceholder.center += c.normal * c.depth;
		}
	}
	movingSphere = spherePlaceholder.center;
	
	if (reRenderText && letters.str().size() > 0)
	{
		reRenderText = false;
		playerTextEntry = fonter.Render(letters.str(), glm::vec4(1, 0, 0, 1));
		std::stringstream().swap(letters);
	}

	const float BulletSpeed = 5.f * timeDelta; //  5 units per second
	spherePlaceholder.radius = BulletRadius;
	Collision c;
	std::vector<DynamicOctTree<PathFollower>::iterator> to_remove;
	for (std::size_t i = 0; i < bullets.size(); i++)
	{
		if (glm::any(glm::greaterThan(glm::abs(bullets[i].position), glm::vec3(20))))
		{
			bullets.erase(bullets.begin() + i);
			i--;
			continue;
		}
		spherePlaceholder.center = bullets[i].position + bullets[i].direction * BulletSpeed;
		for (auto& boxers : staticBoxes.Search(spherePlaceholder.GetAABB()))
		{
			if (boxers->Overlap(spherePlaceholder, c))
			{
				{
					//QuickTimer _timer("New Decal Generation");
					OBB boxed;

					// TODO: investigate
					// TODO: glm::orthonormalize
					//boxed.ReOrient(glm::lookAt(glm::vec3(), bullets[i].direction, glm::vec3(0, 1, 0)));
					glm::mat3 dumb(1.f);
					dumb[0] = glm::normalize(-c.axis);
					if (glm::abs(glm::dot(c.axis, GravityUp)) < 0.85)
					{
						dumb[2] = glm::normalize(glm::cross(dumb[0], GravityUp));
					}
					else
					{
						dumb[2] = glm::normalize(glm::cross(dumb[0], glm::vec3(1, 0, 0)));
					}
					dumb[2] = glm::normalize(glm::cross(dumb[0], bullets[i].direction));
					dumb[1] = glm::normalize(glm::cross(dumb[2], dumb[0]));
					glm::mat4 dumber{ dumb };
					dumber[3] = glm::vec4(0, 0, 0, 1);
					boxed.ReOrient(dumber);
					boxed.ReScale(glm::vec3(gamin.radius * 2.f));
					boxed.ReCenter(c.point - c.axis * BulletSpeed);
					Decal::GetDecal(boxed, staticBoxes, decalVertex);
					decals.BufferData(decalVertex, StaticDraw);
				}
				gamin.center = c.point + c.normal * EPSILON;
				bullets[i].direction = glm::reflect(bullets[i].direction, c.normal);
			}
		}
		Capsule example;
		example.SetTotalLength(2.f);
		for (auto& hit : followers.Search(gamin.GetAABB()))
		{
			example.SetCenter(hit->first.GetPosition());
			if (example.Intersect(gamin))
			{
				to_remove.push_back(hit);
				std::cout << "Hit!" << std::endl;
				gamin.center = glm::vec3(-100.f);
				continue;
			}
		}
		/*
		for (int j = 0; j < followers.size(); j++)
		{
			example.SetCenter(followers[j].GetPosition());
			if (example.Intersect(gamin))
			{
				followers.erase(followers.begin() + j);
				gamin.center = glm::vec3(-100.f);
				j--;
				continue;
			}
		}
		*/
		bullets[i].position = spherePlaceholder.center;
	}
	for (auto& a : to_remove)
	{
		followers.Erase(a);
	}
	//std::cout << "Done with Erasing" << std::endl;
	followed.Update(timeDelta, staticBoxes);
	followers.for_each([timeDelta](auto& a) 
		{
			glm::vec3 old = a.GetPosition();
			a.Update(timeDelta, staticBoxes); 
			return old != a.GetPosition();
		});

	if (debugFlags[DYNAMIC_TREE])
	{
		dynamicTreeBoxes = followers.GetBoxes();
	}
	/*
	for (auto& follow : followers2)
	{
		follow.first.Update(timeDelta, staticBoxes);
	}*/
	Mouse::UpdateEdges();
	help.MouseUpdate();

	skinMats.fill(glm::mat4(1));
	skinMats[0][3] = glm::vec4(0, 0, 0, 1);
	skinMats[1][3] = glm::vec4(-4, 2 * glm::cos(frameCounter / 100.f), 3 * glm::sin(frameCounter * 3 / 100.f), 1);
	glm::mat4 toDie(1);
	toDie[3] = glm::vec4(1, 0, 0, 1);
	toDie = glm::inverse(toDie);
	skinMats[0] = toDie * skinMats[0];
	toDie = glm::mat4(1);
	toDie[3] = glm::vec4(-5, 0, 0, 1);
	toDie = glm::inverse(toDie);
	skinMats[1] = toDie * skinMats[1];

	fonter.GetTextTris(textBuffer, 0, 0, std::format("FPS:{:7.2f}\nTime:{:4.2f}ms\nIdle:{}ns\nDisplay:\n-Concurrent: {}ns\n-GPU Block Time: {}ns\n{} Version\nTest Bool: {}",
		averageFps, 1000.f / averageFps, averageIdle, averageDisplay, averageRender, (featureToggle) ? "New" : "Old", false));

	std::copy(std::begin(keyState), std::end(keyState), std::begin(keyStateBackup));

	const auto endTime = std::chrono::high_resolution_clock::now();
	idleTime = endTime - idleStart;
	lastIdleStart = idleStart;
	// Delay to keep 100 ticks per second idle stuff
	/*
	if (idleTime < std::chrono::milliseconds(10))
	{
		while (std::chrono::high_resolution_clock::displayStartTime() - displayStartTime <= std::chrono::milliseconds(10));
	}
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
	bool state = (action == GLFW_PRESS);

	unsigned char letter = static_cast<unsigned char>(key & 0xFF);

	if (action != GLFW_RELEASE && key < 0xFF)
	{
		unsigned char copied = letter;
		if (std::isalnum(copied))
		{
			if (!(mods & GLFW_MOD_CAPS_LOCK) && mods & GLFW_MOD_SHIFT)
				copied = std::tolower(copied);
			else if (!(mods & GLFW_MOD_SHIFT)) 
				copied = std::tolower(copied);
		}
		letters << copied;
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
		if (key == GLFW_KEY_L)
		{
			struct
			{
				std::array<glm::vec4, 32> points{};
				int length = 32;
				int pad{}, pad2{}, pad3{};
			} point_temp;
			for (auto& p : point_temp.points)
			{
				p = glm::vec4(glm::linearRand(0.f, 1.f), glm::linearRand(0.f, 1.f), 0, 0);
			}
			pointUniformBuffer.BufferData(point_temp, StaticDraw);
			pointUniformBuffer.SetBindingPoint(2);
			pointUniformBuffer.BindUniform();
		}
		if (key == GLFW_KEY_K) shift++;
		if (key == GLFW_KEY_M) cameraPosition.y += 3;
		if (key == GLFW_KEY_N) cameraPosition.y -= 3;
		if (key == GLFW_KEY_LEFT_BRACKET) tessAmount -= 1;
		if (key == GLFW_KEY_RIGHT_BRACKET) tessAmount += 1;
		if (key == GLFW_KEY_Q) glfwSetWindowShouldClose(window, GLFW_TRUE);
		if (key == GLFW_KEY_B) featureToggle = !featureToggle;
		if (key == GLFW_KEY_ENTER) reRenderText = true;
		if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
		{
			std::size_t value = static_cast<std::size_t>(key - GLFW_KEY_0);
			debugFlags[value] = !debugFlags[value];
		}
		if (key >= GLFW_KEY_F1 && key <= GLFW_KEY_F1 + debugFlags.size())
		{
			std::size_t value = static_cast<std::size_t>(key - GLFW_KEY_F1 + 1);
			debugFlags[value] = !debugFlags[value];
		}
	}
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
	glm::vec2 viewPortSize{ Window::Width, Window::Height };
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
	//mouseStatus.buttons = (mouseStatus.buttons & ~(1 << button)) | ((action == GLFW_PRESS) << button);
	Mouse::SetButton(static_cast<Mouse::Button>(button & 0xFF), action == GLFW_PRESS);
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		if (Mouse::CheckButton(Mouse::ButtonRight))
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		}
		else
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !userPortion.Contains(Mouse::GetPosition()))
	{
		glm::mat4 cameraOrientation{};
		Ray liota = GetMouseProjection(Mouse::GetPosition(), cameraOrientation);
		float rayLength = 50.f;

		RayCollision rayd{};
		OBB* point = nullptr;
		for (auto& item : staticBoxes.RayCast(liota))
		{
			if (item->Intersect(liota.initial, liota.delta, rayd) && rayd.depth > 0.f && rayd.depth < rayLength)
			{
				rayLength = rayd.depth;
				point = &(*item);
			}
		}
		// Point displayStartTime has the pointer to the closest element
		Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.1f, rayLength - 0.5f - 0.2f, 30, 30);
		//pointingCapsule.ReOrient(glm::vec3(0, 0, 90.f));
		//pointingCapsule.ReOrient(cameraOrientation);
		//pointingCapsule.ReCenter(cameraPosition);
		//pointingCapsule.Translate(pointingCapsule.Forward() * (0.3f + rayLength / 2.f));
		//pointingCapsule.Rotate(glm::vec3(0, 0, 90.f));
		//pointingCapsule.ReScale(glm::vec3((rayLength - 0.5f) / 2.f, 0.1f, 0.1f));
		bullets.emplace_back<Bullet>({cameraPosition, liota.delta});
	}
	testButton.MouseUpdate();
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && userPortion.Contains(Mouse::GetPosition()))
	{
		userPortion.z -= 25;
	}
}

void mouseCursorFunc(GLFWwindow* window, double xPos, double yPos)
{
	float x = static_cast<float>(xPos), y = static_cast<float>(yPos);
	const glm::vec2 oldPos = Mouse::GetPosition();
	glm::ivec2 deviation = glm::ceil(glm::abs(glm::vec2(Window::Width, Window::Height) / 2.f - oldPos));
	ui_tester = NineSliceGenerate(glm::vec2(Window::Width, Window::Height) / 2.f, deviation);
	ui_tester_buffer.BufferData(ui_tester, StaticDraw);
	Mouse::SetPosition(x, y);

	if (Mouse::CheckButton(Mouse::ButtonRight))
	{
		float xDif = x - oldPos.x;
		float yDif = y - oldPos.y;
		if (abs(xDif) > 20)
			xDif = 0;
		if (abs(yDif) > 20)
			yDif = 0;
		// Why 50??
		float yDelta = 50 * (xDif * ANGLE_DELTA) / Window::Width;
		float zDelta = 50 * (yDif * ANGLE_DELTA) / Window::Height;

		cameraRotation.y += yDelta;
		cameraRotation.x = std::clamp(cameraRotation.x + zDelta, -75.f, 75.f);
	}
	else
	{
		buttonToggle = buttonRect.Contains(x, y);
	}
	if (Mouse::CheckButton(Mouse::ButtonLeft))
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
			float xDif = x - oldPos.x;
			float yDif = y - oldPos.y;
			float yDelta = (xDif * ANGLE_DELTA) / Window::Width;
			float zDelta = -(yDif * ANGLE_DELTA) / Window::Height;


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
			for (auto& foobar : staticBoxes.Search(fum))
			{
				if (foobar->box.Overlap(moveable, slider))
				{
					moveable.ApplyCollision(slider);
				}
			}*/
		}
	}
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
	Window::Width = width;
	Window::Height = height;
	aspectRatio = static_cast<float>(width) / height;
	
	cameraUniformBuffer.Generate(DynamicDraw, 2 * sizeof(glm::mat4));
	cameraUniformBuffer.SetBindingPoint(0);
	cameraUniformBuffer.BindUniform();

	glm::mat4 projection = glm::perspective(glm::radians(Fov * aspectRatio), aspectRatio, zNear, zFar);
	cameraUniformBuffer.BufferSubData(projection, sizeof(glm::mat4));

	depthed.GetColorBuffer<0>().CreateEmpty(Window::Width, Window::Height);
	depthed.GetColorBuffer<0>().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);
	CheckError();
	depthed.GetColorBuffer<1>().CreateEmpty(Window::Width, Window::Height);
	depthed.GetColorBuffer<1>().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);
	CheckError();



	depthed.GetDepth().CreateEmpty(Window::Width, Window::Height, InternalDepth);
	depthed.GetDepth().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);
	CheckError();

	depthed.GetStencil().CreateEmpty(Window::Width, Window::Height, InternalStencil);
	// Doing NearestNearest is super messed up
	depthed.GetStencil().SetFilters(MinNearest, MagNearest, BorderClamp, BorderClamp);
	CheckError();
	depthed.Assemble();

	scratchSpace.GetColorBuffer().CreateEmpty(Window::Width, Window::Height, InternalRGBA);
	scratchSpace.GetColorBuffer().SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);
	scratchSpace.Assemble();

	screenSpaceBuffer.Generate(StaticRead, sizeof(glm::mat4));
	screenSpaceBuffer.SetBindingPoint(1);
	screenSpaceBuffer.BindUniform();
	screenSpaceBuffer.BufferSubData(glm::ortho<float>(0, static_cast<float>(Window::Width), static_cast<float>(Window::Height), 0));
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

int main(int argc, char** argv)
{
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

	windowPointer = glfwCreateWindow(Window::Width, Window::Height, "Wowie a window", nullptr, nullptr);
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
	window_size_callback(nullptr, Window::Width, Window::Height);

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
	std::srand(NULL);
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
	toDisable = 1; // Disable Shader Recompiled due to state change(when you apply a line draw function to something that is expected for tris)
	//glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PERFORMANCE, )

	// TODO: This noise stuff idk man
	//Shader::IncludeInShaderFilesystem("FooBarGamer.gsl", "uniformv.glsl");
	//Shader::IncludeInShaderFilesystem("noise2D.glsl", "noise2D.glsl");

	// SHADER SETUP
	Shader::SetBasePath("Shaders");
	billboardShader.CompileSimple("texture");
	decalShader.CompileSimple("decal");
	dither.CompileSimple("light_text_dither");
	expand.Compile("framebuffer", "expand");
	flatLighting.CompileSimple("lightflat");
	fontShader.CompileSimple("font");
	frameShader.CompileSimple("framebuffer");
	ground.CompileSimple("ground_");
	instancing.CompileSimple("instance");
	nineSlicer.CompileSimple("ui_nine");
	pathNodeView.CompileSimple("path_node");
	skinner.CompileSimple("skin");
	sphereMesh.CompileSimple("mesh");
	stencilTest.CompileSimple("stencil_");
	triColor.CompileSimple("tri_color");
	uiRect.CompileSimple("ui_rect");
	uiRectTexture.CompileSimple("ui_rect_texture");
	uniform.CompileSimple("uniform");
	voronoi.Compile("framebuffer", "voronoi");
	widget.CompileSimple("widget");


	billboardShader.UniformBlockBinding("Camera", 0);
	decalShader.UniformBlockBinding("Camera", 0);
	dither.UniformBlockBinding("Camera", 0);
	flatLighting.UniformBlockBinding("Camera", 0);
	ground.UniformBlockBinding("Camera", 0);
	instancing.UniformBlockBinding("Camera", 0);
	pathNodeView.UniformBlockBinding("Camera", 0);
	skinner.UniformBlockBinding("Camera", 0);
	sphereMesh.UniformBlockBinding("Camera", 0);
	stencilTest.UniformBlockBinding("Camera", 0);
	triColor.UniformBlockBinding("Camera", 0);
	uniform.UniformBlockBinding("Camera", 0);

	nineSlicer.UniformBlockBinding("ScreenSpace", 1);
	uiRect.UniformBlockBinding("ScreenSpace", 1);
	uiRectTexture.UniformBlockBinding("ScreenSpace", 1);
	fontShader.UniformBlockBinding("ScreenSpace", 1);

	voronoi.UniformBlockBinding("Points", 2);

	CheckError();
	// VAO SETUP
	decalVAO.ArrayFormat<TextureVertex>(decalShader);
	fontVAO.ArrayFormat<UIVertex>(fontShader);
	CheckError();
	instanceVAO.ArrayFormat<TextureVertex>(instancing, 0);
	instanceVAO.ArrayFormat<glm::mat4>(instancing, 1, 1);
	instanceVAO.ArrayFormat<TangentVertex>(instancing, 2);
	meshVAO.ArrayFormat<MeshVertex>(sphereMesh);
	CheckError();
	nineSliced.ArrayFormatOverride<glm::vec4>("rectangle", nineSlicer, 0, 1);
	normalVAO.ArrayFormat<NormalVertex>(flatLighting);
	CheckError();
	pathNodeVAO.ArrayFormat<Vertex>(pathNodeView, 0);
	pathNodeVAO.ArrayFormatOverride<glm::vec3>("Position", pathNodeView, 1, 1);
	//normalMapVAO.ArrayFormat<TangentVertex>(instancing, 2);
	CheckError();
	plainVAO.ArrayFormat<Vertex>(uniform);

	texturedVAO.ArrayFormat<TextureVertex>(dither);
	CheckError();
	// TEXTURE SETUP
	// These two textures from https://opengameart.org/content/stylized-mossy-stone-pbr-texture-set, do a better credit
	Texture::SetBasePath("Textures");

	depthMap.Load("depth.png");
	depthMap.SetFilters(LinearLinear, MagLinear, MirroredRepeat, MirroredRepeat);
	depthMap.SetAnisotropy(16.f);

	ditherTexture.Load(dither16, InternalRed, FormatRed, DataUnsignedByte);
	ditherTexture.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	hatching.Load("hatching.png");
	hatching.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	normalMap.Load("normal.png");
	normalMap.SetFilters(LinearLinear, MagLinear, MirroredRepeat, MirroredRepeat);
	normalMap.SetAnisotropy(16.f);

	texture.Load("text.png");
	texture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);

	wallTexture.Load("flowed.png");
	wallTexture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);

	buttonB.CreateEmptyWithFilters(100, 100, InternalRGBA, {}, glm::vec4(0, 1, 1, 1));
	buttonA.CreateEmptyWithFilters(100, 100, InternalRGBA, {}, glm::vec4(1, 0.5, 1, 1));

	nineSlice.Load("9slice.png");
	nineSlice.SetFilters();

	// TODO: Use glm::noise::perlin
	tessMap.Load(tesselationCode, InternalRed, FormatRed, DataUnsignedByte);
	tessMap.SetFilters(LinearLinear, MagLinear);

	/*
	mapper.Generate({ "Textures/skybox/right.jpg", "Textures/skybox/left.jpg", "Textures/skybox/top.jpg",
		"Textures/skybox/bottom.jpg", "Textures/skybox/front.jpg", "Textures/skybox/back.jpg" });
	*/

	stickBuffer.BufferData(stick);
	solidCubeIndex.BufferData(cubeIndicies);

	std::array<TextureVertex, 4> verts{};

	for (int i = 0; i < 4; i++)
		verts[i].position = plane[i];
	verts[0].coordinates = glm::vec2(1, 1);
	verts[1].coordinates = glm::vec2(1, 0);
	verts[2].coordinates = glm::vec2(0, 1);
	verts[3].coordinates = glm::vec2(0, 0);
	
	std::array<TangentVertex, 4> tangents{};
	tangents.fill({ glm::vec3(1, 0, 0), glm::vec3(0, 0, 1) });

	normalMapBuffer.BufferData(tangents);

	texturedPlane.BufferData(verts);

	planeBO.BufferData(plane);

	plainCube.BufferData(plainCubeVerts);

	std::array<glm::vec3, 5> funnys = { {glm::vec3(0.25), glm::vec3(0.5), glm::vec3(2.5, 5, 3), glm::vec3(5, 2, 0), glm::vec3(-5, 0, -3) } };
	pathNodePositions.BufferData(funnys);

	// RAY SETUP
	std::array<glm::vec3, 20> rays = {};
	rays.fill(glm::vec3(0));
	rayBuffer.BufferData(rays);
	// CREATING OF THE PLANES

	for (int i = -5; i <= 5; i++)
	{
		if (abs(i) <= 1)
			continue;
		if (abs(i) == 3)
		{
			GetPlaneSegment(glm::vec3(2 * i, 0, 0), PlusY, instancedModels);
			GetPlaneSegment(glm::vec3(0, 0, 2 * i), PlusY, instancedModels);

			GetPlaneSegment(glm::vec3(2 * i, 0, 2 * i), PlusY, instancedModels);
			GetPlaneSegment(glm::vec3(-2 * i, 0, 2 * i), PlusY, instancedModels);
			for (int x = -2; x <= 2; x++)
			{
				if (x == 0)
					continue;
				GetHallway(glm::vec3(2 * x, 0, 2 * i), instancedModels, false);
				GetHallway(glm::vec3(2 * i, 0, 2 * x), instancedModels, true);
			}
			continue;
		}
		GetHallway(glm::vec3(0, 0, 2 * i), instancedModels, true);
		GetHallway(glm::vec3(2 * i, 0, 0), instancedModels, false);
	}
	for (int i = 0; i < 9; i++)
	{
		GetPlaneSegment(glm::vec3(2 * (i % 3 - 1), 0, 2 * (static_cast<int>(i / 3) - 1)), PlusY, instancedModels);
	}
	// Diagonal Walls
	instancedModels.emplace_back(glm::vec3( 2, 1.f, -2), glm::vec3(0,  45,  90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2))));
	instancedModels.emplace_back(glm::vec3( 2, 1.f,  2), glm::vec3(0, -45,  90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2))));
	instancedModels.emplace_back(glm::vec3(-2, 1.f,  2), glm::vec3(0,  45, -90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2))));
	instancedModels.emplace_back(glm::vec3(-2, 1.f, -2), glm::vec3(0, -45, -90.f), glm::vec3(1, 1, static_cast<float>(sqrt(2))));

	instancedModels.emplace_back(glm::vec3(0.5f, 1, 0), glm::vec3(0, 0, -90.f));
	instancedModels.emplace_back(glm::vec3(0.5f, 1, 0), glm::vec3(0, 0,  90.f));


	// The ramp
	instancedModels.emplace_back(glm::vec3(3.8f, .25f, 0), glm::vec3(0, 0.f, 15.0f), glm::vec3(1, 1, 1));

	// The weird wall behind the player I think?
	instancedModels.emplace_back(glm::vec3(14, 1, 0), glm::vec3(0.f), glm::vec3(1.f, 20, 1.f));

	std::vector<glm::mat4> awfulTemp{};
	awfulTemp.reserve(instancedModels.size());
	//instancedModels.emplace_back(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f));
	for (const auto& ref : instancedModels)
	{
		OBB project(ref);
		glm::vec3 forawrd = project.Up();
		if (glm::dot(forawrd, GravityUp) > 0.5f)
		{
			PathFollower::PathNodes.push_back(PathNode::MakeNode(project.Center() + glm::vec3(0, 1, 0)));
		}
		project.Scale(glm::vec3(1, .0625f, 1));
		project.Scale(glm::vec3(1, 0, 1));
		staticBoxes.Insert(project, project.GetAABB());
		awfulTemp.push_back(ref.GetModelMatrix()); // Because we're using instancedModels to draw them this doesn't have to be the projection for some reason
		instancedDrawOrder.emplace_back<MaxHeapValue<OBB>>({project, 0});
		//awfulTemp.push_back(ref.GetNormalMatrix());
	}
	{
		QuickTimer _tim("Node Culling");
		std::erase_if(PathFollower::PathNodes,
			[&](const PathNodePtr& A)
			{
				AABB boxer{};
				boxer.SetScale(0.75f);
				boxer.Center(A->GetPosition());
				auto temps = staticBoxes.Search(boxer);
				if (temps.size() == 0)
					return false;
				RayCollision fumop{};
				for (auto& temp : temps)
				{
					if (temp->Overlap(boxer))
					{
						return true;
					}
				}
				return false;
			}
		);
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
	CheckError();
	// Decal stuff

	OBB decalGenerator;
	decalGenerator.ReCenter(glm::vec3(1, 0.25, 3));
	decalGenerator.ReScale(glm::vec3(0.5f));
	decalGenerator.Rotate(glm::eulerAngleYZ(glm::radians(-45.f), glm::radians(-26.f)));
	{
		QuickTimer _timer{ "Decal Generation" };
		decals = Decal::GetDecal(decalGenerator, staticBoxes);
	}

	// SKINNING
	std::array<float, 3> dummy{1.f};
	skinBuf.BufferData(dummy, StaticDraw);
	instanceVAO2.ArrayFormat<TextureVertex>(skinner);
	std::vector<GLuint> grs = {
		0, 1, 2, 1, 2, 3,
		2, 3, 4, 3, 4, 5,
		4, 5, 6, 5, 6, 7,
	};
	//std::vector<GLuint> grs = { 0, 2, 3, 1, 0, 3 };
	std::vector<TextureVertex> bosp;
	std::vector<glm::vec3> pogd = {
		glm::vec3(1, 0, 1),
		glm::vec3(1, 0, -1),
		glm::vec3(-1, 0, 1),
		glm::vec3(-1, 0, -1),
		glm::vec3(-3, 0,  1),
		glm::vec3(-3, 0, -1),
		glm::vec3(-5, 0, 1),
		glm::vec3(-5, 0, -1),
	};
	for (auto& a : pogd) { bosp.push_back({ a, glm::vec2() }); }
	skinVertex.BufferData(bosp);

	skinArg.BufferData(grs);

	for (auto& point : verts)
	{
		point.position = glm::mat3(glm::eulerAngleZY(glm::radians(90.f), glm::radians(-90.f))) * point.position;
		//point.position += glm::vec3(0, 1.f, 0);
	}
	billboardBuffer.BufferData(verts);
	constexpr int followsize = 10;
	followers.ReserveSize(followsize);
	for (int i = 0; i < followsize; i++)
	{
		glm::vec2 base = glm::diskRand(20.f);
		//followers.emplace_back(glm::vec3(base.x, 2.5, base.y));
		PathFollower fus(glm::vec3(base.x, 2.5, base.y));
		followers.Insert(fus, fus.GetAABB());
	}
	PathFollower sc(glm::vec3(-100, 2.5, -100));
	PathFollower sc2(glm::vec3(-10, 2.5, -10));
	followers.Insert(sc, sc.GetAABB());
	followers.Insert(sc2, sc2.GetAABB());


	struct 
	{
		std::array<glm::vec4, 32> points{};
		int length = 32;
		int pad{}, pad2{}, pad3{};
	} point_temp;
	for (auto& p: point_temp.points)
	{
		p = glm::vec4(glm::linearRand(0.f, 1.f), glm::linearRand(0.f, 1.f), 0, 0);
	}
	pointUniformBuffer.BufferData(point_temp, StaticDraw);
	pointUniformBuffer.SetBindingPoint(2);
	pointUniformBuffer.BindUniform();

	int maxSize = 0;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxSize);
	// Shenanigans
	depthMap.CreateEmptyWithFilters({ 1024, 1024 }, InternalRed16, { LinearLinear, MagLinear, Repeat, Repeat }, { 0.f, 1.f, 1.f, 1.f });
	//depthMap.Load("depth.png");
	ColorFrameBuffer _t;
	_t.GetColor().MakeAliasOf(depthMap);
	_t.Assemble();
	_t.Bind();
	glViewport(0, 0, 1024, 1024);
	voronoi.SetActiveShader();
	voronoi.SetInt("mode", 2);
	voronoi.DrawArray<DrawType::TriangleStrip>(4);
	BindDefaultFrameBuffer();
	depthMap.BindTexture();
	//depthMap.SetAnisotropy(16.f);

	HeightToNormal(depthMap, normalMap);
	normalMap.BindTexture();
	normalMap.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);
	normalMap.SetAnisotropy(16.f);



	// =============================================================
	// Pathfinding stuff
	std::vector<glm::vec3> boxingDay{};
	std::vector<glm::vec3> littleTrolling{};
	{
		QuickTimer _timer("Node Connections");
		for (std::size_t i = 0; i < PathFollower::PathNodes.size(); i++)
		{
			for (std::size_t j = i + 1; j < PathFollower::PathNodes.size(); j++)
			{
				PathNode::addNeighbor(PathFollower::PathNodes[i], PathFollower::PathNodes[j],
					[&](const PathNodePtr& A, PathNodePtr& B)
					{
						glm::vec3 a = A->GetPosition(), b = B->GetPosition();
						float delta = glm::length(a - b);
						if (delta > 5.f) // TODO: Constant
							return false;
						Ray liota(a, b - a);
						auto temps = staticBoxes.RayCast(liota);
						if (temps.size() == 0)
							return true;
						RayCollision fumop{};
						for (auto& temp : temps)
						{
							if (temp->Intersect(liota.initial, liota.direction, fumop) && fumop.depth < delta)
							{
								return false;
							}
						}
						return true;
					}
				);
			}
		}
		// TODO: Second order check to remove connections that are "superfluous", ie really similar in an unhelpful manner
		std::erase_if(PathFollower::PathNodes, [](const PathNodePtr& A) {return A->neighbors().size() == 0; });
		for (std::size_t i = 0; i < PathFollower::PathNodes.size(); i++)
		{
			auto& local = PathFollower::PathNodes[i];
			auto localBoys = local->neighbors();
			for (std::size_t j = 0; j < localBoys.size(); j++)
			{
				PathNodePtr weaker;
				if ((weaker = localBoys[j].lock()))
				{
					glm::vec3 deltaA = weaker->GetPosition() - local->GetPosition();
					for (std::size_t k = j; k < localBoys.size(); k++)
					{
						PathNodePtr weakest;
						if ((weakest = localBoys[k].lock()))
						{
							glm::vec3 deltaB = weakest->GetPosition() - local->GetPosition();
							// This sucks
						}
					}
				}
			}
		}
	}


	
	for (auto& autod : PathFollower::PathNodes)
	{
		boxingDay.push_back(autod->GetPosition());
		for (auto& weak : autod->neighbors())
		{
			auto weaker = weak.lock();
			if (weaker)
			{
				littleTrolling.push_back(autod->GetPosition());
				littleTrolling.push_back(weaker->GetPosition());
			}
		}
	}
	
	pathNodePositions.BufferData(boxingDay, StaticDraw);
	pathNodeLines.BufferData(littleTrolling, StaticDraw);

	// =============================================================

	{
		QuickTimer _tim{ "Sphere/Capsule Generation" };
		Sphere::GenerateMesh(sphereBuffer, sphereIndicies, 30, 30);
		Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.1f, 10.f, 30, 30);
		Capsule::GenerateMesh(movingCapsule, movingCapsuleIndex, 0.25f, 0.5f, 30, 30);
	}

	//OBJReader::ReadOBJ("Models\\rock.obj", sphereBuffer, sphereIndicies);

	catapult.SetCenter(glm::vec3(0, 0.5f, 0));
	catapult.SetRadius(0.25f);
	catapult.SetLineLength(0.5f);

	catapultModel.translation = glm::vec3(0, 0.5f, 0);
	catapultBox.ReCenter(glm::vec3(0, 0.5, 0));
	catapultBox.ReScale(glm::vec3(0.25f, 0.5f, 0.25f));


	Font::SetFontDirectory("Fonts");

	// Awkward syntax :(

	{
		QUICKTIMER("Font Loading");
		ASCIIFont::LoadFont(fonter, "CommitMono-400-Regular.ttf", 25.f, 2, 2);
	}

	stickIndicies.BufferData(stickDex, StaticDraw);

	cubeOutlineIndex.BufferData(cubeOutline, StaticDraw);

	dumbBox.ReCenter(glm::vec3(0, 1.f, -2));
	dumbBox.Scale(glm::vec3(1.f));
	dumbBox.Rotate(glm::vec3(0, -90, 0));

	moveable.ReCenter(glm::vec3(0, .25, 0));
	moveable.Scale(0.25f);

	pointingCapsule.ReCenter(glm::vec3(0, 5, 0));
	pointingCapsule.ReOrient(glm::vec3(0.f, 0, 90.f));
	//staticBoxes.Insert({ dumbBox, false }, dumbBox.GetAABB());
	//Log("Doing it");
	//windowResize(1000, 1000);
	Button buttonMan({ 0, 0, 20, 20 }, Dumber);
	
	std::array<std::string, 2> buttonText{ "Soft", "Not" };

	Texture2D tempA, tempB;
	fonter.RenderToTexture(tempA, "Soft", glm::vec4(0, 0, 0, 1));
	fonter.RenderToTexture(tempB, "Not", glm::vec4(0, 0, 0, 1));

	ColorFrameBuffer buffered;
	glm::ivec2 bufSize = glm::max(tempA.GetSize(), tempB.GetSize()) + glm::ivec2(20);
	auto sized = NineSliceGenerate(glm::ivec2(0, 0), bufSize);
	screenSpaceBuffer.Generate(StaticRead, sizeof(glm::mat4));
	screenSpaceBuffer.SetBindingPoint(1);
	screenSpaceBuffer.BindUniform();
	screenSpaceBuffer.BufferSubData(glm::ortho<float>(0, static_cast<float>(bufSize.x), static_cast<float>(bufSize.y), 0));
	glViewport(0, 0, bufSize.x, bufSize.y);
	EnableGLFeatures<Blending>();
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	ArrayBuffer rects;
	rects.BufferData(sized, StaticDraw);

	for (int j = 0; j < 2; j++)
	{
		auto& current = (j == 0) ? buttonA : buttonB;
		current.CreateEmpty(bufSize.x, bufSize.y);
		buffered.GetColor().MakeAliasOf(current);
		buffered.Assemble();
		buffered.Bind();
		nineSlicer.SetActiveShader();
		nineSliced.BindArrayBuffer(rects);
		nineSlicer.SetTextureUnit("image", nineSlice);
		nineSlicer.DrawArrayInstanced<DrawType::TriangleStrip>(4, 9);
		uiRectTexture.SetActiveShader();
		uiRectTexture.SetVec4("rectangle", glm::vec4(0, 0, bufSize));
		uiRectTexture.SetTextureUnit("image", (j == 0) ? tempA : tempB, 0);
		uiRectTexture.DrawArray<DrawType::TriangleStrip>(4);
	}
	CheckError();
	DisableGLFeatures<Blending>();
	glLineWidth(100);

	help.SetMessages("Work", "UnWork", fonter);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glClearColor(0.f, 0.f, 0.f, 0.f);
	Log("End of Init");
}