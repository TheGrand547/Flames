#include <chrono>
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <freeglut.h>
#include <iostream>
#include <map>
#include <sys/utime.h>
#include <time.h>
#include "AABB.h"
#include "Buffer.h"
#include "CubeMap.h"
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
Buffer<ArrayBuffer> stickBuffer, texturedPlane, plainCube, planeBO, rayBuffer, albertBuffer, sphereBuffer;
Buffer<ElementArray> sphereIndicies, stickIndicies, cubeOutlineIndex;

UniformBuffer universal;

// Shaders
Shader dither, expand, finalResult, frameShader, flatLighting, uniform, sphereMesh;

// Textures
Texture2D ditherTexture, framebufferColor, framebufferDepth, framebufferNormal, hatching, normalModifier, texture, wallTexture;
CubeMap mapper;

// Vertex Array Objects
VAO texturedVAO, normalVAO, plainVAO, meshVAO;


// TODO: Make structures around these
GLuint framebuffer, framebufferMod;


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

void display()
{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	GLenum buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, buffers);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	uniform.SetActiveShader();
	//glm::mat4 projection = glm::perspective(glm::radians(70.f), 1.f, zNear, zFar);

	// Camera matrix
	glm::vec3 angles2 = glm::radians(cameraRotation);

	// Adding pi/2 is necessary because the default camera is facing -z
	glm::mat4 view = glm::translate(glm::eulerAngleXYZ(angles2.x, angles2.y + glm::half_pi<float>(), angles2.z), -cameraPosition);
	universal.BufferSubData(view, 0);

	dither.SetActiveShader();
	glm::vec3 colors(.5f, .5f, .5f);
	dither.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	dither.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	dither.SetVec3("viewPos", cameraPosition);
	dither.SetTextureUnit("textureIn", wallTexture, 0);
	dither.SetTextureUnit("ditherMap", ditherTexture, 1);

	texturedVAO.BindArrayBuffer(texturedPlane);

	for (Model& model : planes)
	{
		glm::vec3 color(.5f, .5f, .5f);
		dither.SetMat4("Model", model.GetModelMatrix());
		dither.SetVec3("color", color);
		dither.DrawElements<TriangleStrip>(texturedPlane);
	}

	/* STICK FIGURE GUY */
	uniform.SetActiveShader();
	plainVAO.BindArrayBuffer(stickBuffer);

	colors = glm::vec3(1, 0, 0);
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
			glLineWidth((box.color) ? wid * 1.5f : wid);
			glPointSize((box.color) ? wid * 1.5f : wid);
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
	uniform.DrawIndexed<Lines>(cubeOutlineIndex);

	// Albert
	texturedVAO.BindArrayBuffer(albertBuffer);
	dither.SetActiveShader();
	dither.SetTextureUnit("ditherMap", wallTexture, 1);
	dither.SetTextureUnit("textureIn", texture, 0);
	dither.SetMat4("Model", smartBox.GetModelMatrix());
	dither.SetVec3("color", (!smartBoxColor) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0));
	dither.DrawElements<Triangle>(albertBuffer);

	// Drawing of the rays
	glDisable(GL_DEPTH_TEST);
	plainVAO.BindArrayBuffer(rayBuffer);
	Model bland;
	uniform.SetMat4("Model", bland.GetModelMatrix());
	uniform.SetVec3("color", glm::vec3(0.7f));
	//uniform.DrawElements<Lines>(rayBuffer);
	glPointSize(15.f);
	uniform.DrawElements<Points>(rayBuffer);
	glEnable(GL_DEPTH_TEST);

	// Sphere drawing
	glEnable(GL_CULL_FACE);
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
	sphereMesh.DrawIndexed(Triangle, sphereIndicies);
	// Calling with triangle_strip is fucky
	/*
	flatLighting.DrawIndexed(Triangle, sphereIndicies);
	sphereModel.translation = moveSphere;
	flatLighting.SetMat4("modelMat", sphereModel.GetModelMatrix());
	flatLighting.SetMat4("normMat", sphereModel.GetNormalMatrix());
	flatLighting.DrawIndexed(Triangle, sphereIndicies);
	*/

	// Framebuffer stuff
	CheckError();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferMod);
	glDrawBuffers(1, buffers);
	glClear(GL_COLOR_BUFFER_BIT);
	
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);;
	CheckError();
	frameShader.SetActiveShader();
	frameShader.SetTextureUnit("normal", framebufferNormal, 0);
	frameShader.SetTextureUnit("depth", framebufferDepth, 1);
	frameShader.SetFloat("zNear", zNear);
	frameShader.SetFloat("zFar", zFar);
	frameShader.SetInt("zoop", kernel);

	frameShader.DrawElements<TriangleStrip>(4);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	expand.SetActiveShader();
	expand.SetTextureUnit("screen", framebufferColor, 0);
	expand.SetTextureUnit("edges", normalModifier, 1);
	expand.SetTextureUnit("depths", framebufferDepth, 2);
	expand.SetInt("depth", 5);
	frameShader.DrawElements<TriangleStrip>(4);
	
	glFlush();
	glutSwapBuffers();
	CheckError();
}

bool smartBoxCollide(int depth = 0)
{
	if (depth > 4)
		return true;
	bool val = false;
	for (auto& letsgo : boxes.Search(smartBox.GetAABB()))
	{
		Collision c;
		if (smartBox.Overlap(letsgo->box, c))
		{
			std::array<glm::vec3, 2> pointerss = { c.point, c.point + c.normal };
			rayBuffer.BufferSubData(pointerss);
			val = true;
			smartBox.OverlapWithResponse(letsgo->box);
		}
	}
	// TODO: Draw the lines and things y'know
	smartBox.OverlapWithResponse(dumbBox);
	//dumbBox.OverlapWithResponse(smartBox);

	return val;
}

void idle()
{
	static auto lastTimers = std::chrono::high_resolution_clock::now();
	frameCounter++;
	const auto now = std::chrono::high_resolution_clock::now();
	const auto delta = now - lastTimers;

	OBB goober2(AABB(glm::vec3(0), glm::vec3(1)));
	goober2.Translate(glm::vec3(2, 0.1, 0));	
	goober2.Rotate(glm::radians(glm::vec3(0, frameCounter * 4.f, 0)));
	glm::mat4 tester = goober2.GetNormalMatrix();
	std::cout << "\33[2K\r" << std::chrono::duration_cast<std::chrono::microseconds>(delta).count() << "\t" << std::chrono::duration<float, std::chrono::seconds::period>(delta).count();

	//std::cout << "\r" << goober2.Forward() << "\t" << goober2.Cross() << "\t" << goober2.Up();
	//std::cout << "\r" << "AABB Axis: " << goober2.Forward() << "\t Euler Axis" << tester * glm::vec4(1, 0, 0, 0) << std::endl;
	//std::cout << "\r" << "AABB Axis: " << goober2.Forward() << "\t Euler Axis" << glm::transpose(tester)[0];
	//std::cout << "\r" << (float)elapsed / 1000.f << "\t" << 1000.f / float(elapsed) << "\t" << kernel << "\t" << lineWidth;
	Plane foobar(glm::vec3(1, 0, 0), glm::vec3(4, 0, 0)); // Facing away from origin
	//foobar.ToggleTwoSided();
	//if (!smartBox.IntersectionWithResponse(foobar))
		//sacounter++;
		//std::cout << counter << std::endl;
	//smartBox.RotateAbout(glm::vec3(0.05f, 0.07f, -0.09f), glm::vec3(0, -5, 0));
	//smartBox.RotateAbout(glm::vec3(0, 0, 0.05f), glm::vec3(0, -2, 0));
	//smartBox.RotateAbout(glm::vec3(0.05f, 0, 0), glm::vec3(0, -5, 0));

	//float speed = 3 * ((float) elapsed) / 1000.f;
	float speed = 3 * std::chrono::duration<float, std::chrono::seconds::period>(delta).count();

	glm::vec3 forward = glm::eulerAngleY(glm::radians(-cameraRotation.y)) * glm::vec4(1, 0, 0, 0);
	glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
	forward = speed * glm::normalize(forward);
	right = speed * glm::normalize(right);
	glm::vec3 previous = cameraPosition;
	if (keyState[ArrowKeyUp])
	{
		smartBox.Translate(smartBox.Forward() * speed);
		//moveSphere += smartBox.Forward() * speed;
	}
	if (keyState[ArrowKeyDown])
	{
		smartBox.Translate(smartBox.Forward() * -speed);
		//moveSphere -= smartBox.Forward() * speed;
	}
	if (keyState[ArrowKeyRight]) smartBox.Rotate(glm::vec3(0, -1.f, 0));
	if (keyState[ArrowKeyLeft])  smartBox.Rotate(glm::vec3(0, 1.f, 0));
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
	if (keyState['k'])
		cameraPosition.y = -10;
	if (keyState['b'])
		cameraPosition.y = 10;
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
				playerOb.OverlapWithResponse(wall.box);
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
	
	smartBoxColor = smartBoxCollide();

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
	smartBox.ReOrient(glm::vec3(0, 0, 0));
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
	if (key == 'r' || key == 'R') smartReset();
	if (key >= '1' && key <= '9')
	{
		std::size_t value = (std::size_t) key - '0';
		debugFlags[value] = !debugFlags[value];
	}
	if (key == 'r' || key == 'R')
	{
		glm::vec3 angles2 = glm::radians(cameraRotation);

		glm::vec3 gamer = glm::normalize(glm::eulerAngleXYZ(-angles2.z, -angles2.y, -angles2.x) * glm::vec4(1, 0, 0, 0));
		std::array<glm::vec3, 8> verts = { cameraPosition, cameraPosition + gamer * 100.f , cameraPosition, cameraPosition + gamer * 100.f };
		bool set = false;

		/*
		Collision nears, fars;
		smartBox.Intersect(cameraPosition, gamer, nears, fars);
		auto foosball = smartBox.ClosestFacePoints(cameraPosition);
		std::array<glm::vec3, 12> localpoints;
		for (std::size_t i = 0; i < localpoints.size() && i < foosball.size(); i++)
		{
			localpoints[i] = foosball[i];
		}
		rayBuffer.BufferSubData(localpoints);
		*/
		//for (std::size_t i = 0; i < boxes.size(); i++)
		/*
		for (auto& box: boxes)
		{
			//boxColor[i] = boxes[i].Intersect(offset, gamer * 100.f, nears, fars);
			box.color = box.box.Intersect(offset, gamer * 100.f, nears, fars);q
			if (box.color && !set)
			{
				set = true;
				glm::vec3 point = offset + gamer * nears.distance * 100.f;
				for (std::size_t j = 0; j < 3; j++)
				{
					verts[2 + 2 * j] = point;
					glm::vec3 cur = glm::normalize(box.box[j]);
					verts[2 + 2 * j + 1] = point + SlideAlongPlane(cur, gamer) * 100.f;//point + glm::normalize(gamer - glm::dot(gamer, cur) * cur) * 100.f;
				}
			}
		}*/
		//rayBuffer.BufferSubData(verts);
	}
}

void keyboardOff(unsigned char key, int x, int y)
{
	keyState[key] = false;
}

void mouseFunc(int x, int y)
{
	static int previousX, previousY;
	float xDif = (float) (x - previousX);
	float yDif = (float) (y - previousY);
	if (abs(xDif) > 20)
		xDif = 0;
	if (abs(yDif) > 20)
		yDif = 0;
	float yDelta = 50 * (xDif * ANGLE_DELTA) / glutGet(GLUT_WINDOW_WIDTH);
	float zDelta = 50 * (yDif * ANGLE_DELTA) / glutGet(GLUT_WINDOW_HEIGHT);

	cameraRotation.x += zDelta;
	cameraRotation.y += yDelta;

	previousX = x;
	previousY = y;
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

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	//glDisable(GL_LINE_SMOOTH);
	//glDisable(GL_POLYGON_SMOOTH);

	glDepthFunc(GL_LESS);
	glClearColor(0, 0, 0, 1);
	glLineWidth(5);

	glFrontFace(GL_CCW);

	glutDisplayFunc(display);
	glutIdleFunc(idle);

	glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardOff);
	glutSpecialFunc(specialKeys);
	glutSpecialUpFunc(specialKeysUp);

	glutMotionFunc(mouseFunc);
	glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);


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
	Shader::SetBasePath("Shaders/");
	// TODO: texture loading base path thingy

	uniform.CompileSimple("uniform");
	dither.CompileSimple("light_text_dither");
	flatLighting.CompileSimple("lightflat");


	frameShader.CompileSimple("framebuffer");
	sphereMesh.CompileSimple("mesh");

	texture.Load("Textures/text.png");
	wallTexture.Load("Textures/flowed.png");
	texture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);
	wallTexture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);
	CheckError();

	/*
	mapper.Generate({ "Textures/skybox/right.jpg", "Textures/skybox/left.jpg", "Textures/skybox/top.jpg",
		"Textures/skybox/bottom.jpg", "Textures/skybox/front.jpg", "Textures/skybox/back.jpg" });
	*/
	stickBuffer.Generate();
	stickBuffer.BufferData(stick, StaticDraw);

	plainVAO.Generate();
	plainVAO.FillArray<Vertex>(uniform);


	std::array<TextureVertex, 4> verts{};
	for (int i = 0; i < 4; i++)
		verts[i].position = plane[i];
	verts[0].coordinates = glm::vec2(1, 1);
	verts[1].coordinates = glm::vec2(1, 0);
	verts[2].coordinates = glm::vec2(0, 1);
	verts[3].coordinates = glm::vec2(0, 0);
	texturedPlane.Generate();
	texturedPlane.BufferData(verts, StaticDraw);

	texturedVAO.Generate();
	texturedVAO.FillArray<TextureVertex>(dither);

	planeBO.Generate();
	planeBO.BufferData(plane, StaticDraw);
	CheckError();

	plainCube.Generate();
	plainCube.BufferData(plainCubeVerts, StaticDraw);

	CheckError();

	std::array<glm::vec3, 32> gobs = {glm::vec3(4, 1, -1), glm::vec3(4, 1, 1)};
	gobs.fill(glm::vec3(0, -5, 0));
	gobs[0] += glm::vec3(1, 0, 0);
	gobs[1] += glm::vec3(-1, 0, 0);
	gobs[2] += glm::vec3(0, 0, 1);
	gobs[3] += glm::vec3(0, 0, -1);
	rayBuffer.Generate();
	rayBuffer.BufferData(gobs, StaticDraw);

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

	// Slope
	planes.push_back(Model(glm::vec3(11.8f, .5f, 0), glm::vec3(0, 0.f, 25.0f), glm::vec3(1, 1, 1)));

	//planes.push_back(Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f)));
	for (const auto& ref : planes)
	{
		OBB project(ref);
		project.Scale(glm::vec3(1, .0625f, 1));
		boxes.Insert({project, false}, project.GetAABB());
	}
	Model oops = planes[planes.size() / 2 + 1];

	ditherTexture.Load(dither16, InternalRed, FormatRed, DataUnsignedByte);
	CheckError();

	ditherTexture.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);
	CheckError();


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

	// Framebuffer stuff
	framebufferColor.CreateEmpty(1000, 1000, InternalRGB);
	framebufferColor.SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	framebufferDepth.CreateEmpty(1000, 1000, InternalDepth);
	framebufferDepth.SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	framebufferNormal.CreateEmpty(1000, 1000, InternalRGB);
	framebufferNormal.SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	normalModifier.CreateEmpty(1000, 1000, InternalRGBA);
	normalModifier.SetFilters(MinLinear, MagLinear, BorderClamp, BorderClamp);

	// TODO: Framebuffer class to do this stuff
	// TODO: Renderbuffer for buffers that don't need to be directly read
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferColor.GetGLTexture(), 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, framebufferNormal.GetGLTexture(), 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, framebufferDepth.GetGLTexture(), 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "Framebuffer incomplete ahhhhh" << std::endl;
		exit(-1);
	}


	glGenFramebuffers(1, &framebufferMod);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferMod);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normalModifier.GetGLTexture(), 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "Framebuffer incomplete ahhhhh" << std::endl;
		exit(-1);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	expand.Compile("framebuffer", "expand");


	GenerateSphereMesh(sphereBuffer, sphereIndicies, 30, 30);
	CheckError();
	meshVAO.Generate();
	CheckError();
	meshVAO.FillArray<MeshVertex>(sphereMesh);

	normalVAO.Generate();
	CheckError();
	normalVAO.FillArray<NormalVertex>(flatLighting);

	CheckError();

	stickIndicies.Generate();
	stickIndicies.BufferData(stickDex, StaticDraw);

	CheckError();

	cubeOutlineIndex.Generate();
	cubeOutlineIndex.BufferData(cubeOutline, StaticDraw);

	CheckError();

	hatching.Load("Textures/hatching.png");
	hatching.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	CheckError();

	uniform.UniformBlockBinding("Camera", 0);
	dither.UniformBlockBinding("Camera", 0);
	flatLighting.UniformBlockBinding("Camera", 0);
	sphereMesh.UniformBlockBinding("Camera", 0);

	CheckError();

	smartBox.Scale(glm::vec3(0.5f));
	smartReset();
	dumbBox.ReCenter(glm::vec3(0, 0.75f, -2));
	dumbBox.Scale(glm::vec3(0.5f));
	dumbBox.Rotate(glm::vec3(0, 180, 0));

	CheckError();

	universal.Generate(DynamicDraw, 2 * sizeof(glm::mat4));
	universal.SetBindingPoint(0);
	universal.BindUniform();

	glm::mat4 projection = glm::perspective(glm::radians(70.f), 1.f, zNear, zFar);
	universal.BufferSubData(projection, sizeof(glm::mat4));
	CheckError();

	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	//glClearColor(1.f, 1.f, 1.f, 1.f);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	CheckError();
	glutMainLoop();

	glDeleteFramebuffers(1, &framebuffer);
	return 0;
}