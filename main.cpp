#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <freeglut.h>
#include <iostream>
#include <map>
#include "AABB.h"
#include "Buffer.h"
#include "Shader.h"
#include "glmHelp.h"
#include "Lines.h"
#include "log.h"
#include "Model.h"
#include "OrientedBoundingBox.h"
#include "Plane.h"
#include "StaticOctTree.h"
#include "stbWrangler.h"
#include "Sphere.h"
#include "Texture2D.h"
#include "UniformBuffer.h"
#include "util.h"
#include "Vertex.h"
#include "VertexArray.h"
#include "Wall.h"

template <class T> inline void CombineVector(std::vector<T>& left, const std::vector<T>& right)
{
	left.insert(left.end(), std::make_move_iterator(right.begin()), std::make_move_iterator(right.end()));
}

// Cringe globals
GLuint triVBO, planeBO, cubeIndex, aabbVAO;
Shader uniform;
Buffer buffer;

UniformBuffer universal;

VAO gamerTest, vertexVAO, sphereVAO;

GLuint sphereBuf, sphereIndex, sphereCount;
Shader sphereShader;

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
		{-1, -1, -1},
		{ 1, -1, -1},
		{ 1,  1, -1},
		{-1,  1, -1},
		{-1, -1,  1},
		{ 1, -1,  1},
		{ 1,  1,  1},
		{-1,  1,  1},
	}
};

static const std::array<GLubyte, 36> cubeIndicies =
{
	0, 1, 4,
	1, 5, 4,

	4, 3, 0,
	3, 4, 7,

	7, 4, 6,
	4, 5, 6,

	6, 5, 2,
	2, 5, 1,

	2, 1, 0,
	2, 0, 3,

	6, 2, 7,
	2, 3, 7
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

glm::vec3 plane[] =
{
	{ 1, 0,  1},
	{ 1, 0, -1},
	{-1, 0,  1},
	{-1, 0, -1}
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

GLubyte stickDex[] = { 0, 2, 1, 2, 4, 5, 4, 6, 4, 3, 8, 7, 9, 3 };
GLuint stickBuf;// , stickVAO;
VAO stickVAO;

GLubyte planeOutline[] = { 0, 1, 3, 2, 0 };
Texture2D texture, wallTexture;

bool outlineBoxes = false;

glm::vec3 offset(0, 1.5f, 0);
glm::vec3 angles(0, 0, 0);

GLuint texturedPlane, texturedVAO;

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

Texture2D ditherTexture;
Shader dither;


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

GLuint framebuffer, framebufferMod, frameVAO; 
Texture2D framebufferColor, framebufferDepth, framebufferNormal, normalModifier;
Buffer framebufferBuffer;

Shader expand, finalResult;

// I do not like this personally tbqh
static std::array<glm::vec4, 6> FrameBufferVerts = {
{
	{-1.0f,  1.0f, 0.0f, 1.0f},
	{-1.0f, -1.0f, 0.0f, 0.0f},
	{ 1.0f, -1.0f, 1.0f, 0.0f},
	{-1.0f,  1.0f, 0.0f, 1.0f},
	{ 1.0f, -1.0f, 1.0f, 0.0f},
	{ 1.0f,  1.0f, 1.0f, 1.0f}
}};
Shader frameShader;
Texture2D hatching;

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
std::vector<Model> planes;

struct Dummy
{
	OBB box;
	bool color;
};

//std::vector<OBB> boxes;
StaticOctTree<Dummy> boxes(glm::vec3(20));
std::vector<bool> boxColor;

bool dummyFlag = false;
bool clear = false;
static int counter = 0;

Buffer rayBuffer;
VAO rayVAO;

OBB smartBox;
bool smartBoxColor = false;

//StaticOctTree<glm::vec3> tree;

void display()
{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	GLenum buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, buffers);
	CheckError();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// FORWARD IS (0, 0, 1)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	uniform.SetActiveShader();
	vertexVAO.BindArrayObject();
	glm::mat4 projection = glm::perspective(glm::radians(70.f), 1.f, 0.1f, 100.0f);

	// Camera matrix
	glm::vec3 angles2 = glm::radians(angles);

	// Adding pi is necessary because the default camera is facing -z
	glm::mat4 view = glm::translate(glm::eulerAngleXYZ(angles2.x, angles2.y + glm::half_pi<float>(), angles2.z), -offset);
	universal.BufferSubData(view, 0);

	dither.SetActiveShader();
	wallTexture.BindTexture(0);
	ditherTexture.BindTexture(1);

	glBindBuffer(GL_ARRAY_BUFFER, texturedPlane);
	glm::vec3 colors(.5f, .5f, .5f);
	dither.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	dither.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	dither.SetVec3("viewPos", offset);
	dither.SetTextureUnit("textureIn", 0);
	dither.SetTextureUnit("ditherMap", 1);

	gamerTest.BindArrayObject();

	for (Model& model : planes)
	{
		glm::vec3 color(.5f, .5f, .5f);
		dither.SetMat4("Model", model.GetModelMatrix());
		dither.SetVec3("color", color);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}
	uniform.SetActiveShader();
	stickVAO.BindArrayObject();
	colors = glm::vec3(1, 0, 0);
	Model m22(glm::vec3(10, 0, 0));
	uniform.SetMat4("Model", m22.GetModelMatrix());
	uniform.SetVec3("color", colors);
	glDrawElements(GL_LINE_STRIP, sizeof(stickDex), GL_UNSIGNED_BYTE, stickDex);

	if (outlineBoxes)
	{
		glm::vec3 blue(0, 0, 1);
		glBindVertexArray(aabbVAO);

		OBB goober(AABB(glm::vec3(0), glm::vec3(1)));
		goober.Translate(glm::vec3(2, 0.1, 0));
		goober.Rotate(glm::radians(glm::vec3(0, counter * 4.f, 0)));
		uniform.SetMat4("Model", goober.GetModel().GetModelMatrix());
		uniform.SetVec3("color", blue);

		float wid = 10;
		//glGetFloatv(GL_LINE_WIDTH, &wid);
		glDrawElements(GL_LINES, (GLuint) cubeOutline.size(), GL_UNSIGNED_BYTE, cubeOutline.data());
		//for (std::size_t i = 0; i < boxes.size(); i++)
		for (const auto& box: boxes)
		{
			uniform.SetMat4("Model", box.box.GetModel().GetModelMatrix());
			uniform.SetVec3("color", (box.color) ? colors : blue);
			glLineWidth((box.color) ? wid * 1.5f : wid);
			glPointSize((box.color) ? wid * 1.5f : wid);
			glDrawElements(GL_LINES, (GLuint) cubeOutline.size(), GL_UNSIGNED_BYTE, cubeOutline.data());
			glDrawArrays(GL_POINTS, 0, 8);
		}
		glLineWidth(wid);
		glPointSize(wid);
		uniform.SetMat4("Model", smartBox.GetModel().GetModelMatrix());
		uniform.SetVec3("color", (!smartBoxColor) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0));
		glDrawElements(GL_LINES, (GLuint)cubeOutline.size(), GL_UNSIGNED_BYTE, cubeOutline.data());
		glDrawArrays(GL_POINTS, 0, 8);
		glEnable(GL_CULL_FACE);
	}

	rayVAO.BindArrayObject();
	Model bland;
	uniform.SetMat4("Model", bland.GetModelMatrix());
	uniform.SetVec3("color", glm::vec3(0, 0, 0));
	glDrawArrays(GL_LINES, 0, 8);

	glEnable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	sphereShader.SetActiveShader();
	sphereVAO.BindArrayObject();
	Model sphereModel(glm::vec3(6.5f, 1.5f, 0.f));
	sphereModel.scale = glm::vec3(0.5f);

	
	hatching.BindTexture(0);
	sphereShader.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	sphereShader.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	sphereShader.SetVec3("viewPos", offset);
	sphereShader.SetVec3("shapeColor", glm::vec3(1.f, .75f, 0.f));
	sphereShader.SetMat4("modelMat", sphereModel.GetModelMatrix());
	sphereShader.SetMat4("normMat", sphereModel.GetNormalMatrix());
	sphereShader.SetTextureUnit("hatching", 0);

	// Doing this while letting the normal be the color will create a cool effect
	//glDrawArrays(GL_TRIANGLES, 0, 1836);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIndex);
	// Calling with triangle_strip is fucky
	glDrawElements(GL_TRIANGLES, sphereCount, GL_UNSIGNED_INT, nullptr);
	sphereModel.translation = glm::vec3(0, 1.5f, 6.5f);
	sphereShader.SetMat4("modelMat", sphereModel.GetModelMatrix());
	sphereShader.SetMat4("normMat", sphereModel.GetNormalMatrix());
	glDrawElements(GL_TRIANGLES, sphereCount, GL_UNSIGNED_INT, nullptr);

	// Framebuffer stuff
	CheckError();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferMod);
	glDrawBuffers(1, buffers);
	glClear(GL_COLOR_BUFFER_BIT);
	
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	framebufferNormal.BindTexture(0);
	//framebufferDepth.Bind(0);
	CheckError();
	frameShader.SetActiveShader();
	frameShader.SetTextureUnit("normal", 0);
	glBindVertexArray(frameVAO);
	CheckError();
	glDrawArrays(GL_TRIANGLES, 0, 6);
	CheckError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	framebufferColor.BindTexture(0);
	normalModifier.BindTexture(1);
	framebufferDepth.BindTexture(2);
	expand.SetActiveShader();
	expand.SetTextureUnit("screen", 0);
	expand.SetTextureUnit("edges", 1);
	expand.SetTextureUnit("depths", 2);
	expand.SetInt("depth", 2);
	glBindVertexArray(frameVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	glFlush();
	glutSwapBuffers();
	CheckError();
}

#define ArrowKeyUp    0
#define ArrowKeyDown  1
#define ArrowKeyRight 2
#define ArrowKeyLeft  3

std::vector<bool> keyState(UCHAR_MAX);
std::vector<bool> keyStateBackup(UCHAR_MAX);
std::vector<Wall> walls;

// To get a perpendicular vector to a vector <a, b, c> do that cross <1, 0, 0> to get <0, c, -b>

glm::vec3 rayStart, rayDir;

bool smartBoxCollide(glm::vec3 forward, int depth = 0)
{
	if (depth > 4)
		return true;
	auto gamerse = boxes.Search(smartBox.GetAABB());
	std::cout << "Depth: " << depth << "\t Count: " << gamerse.size() << std::endl;
	for (auto& letsgo : gamerse)
	{
		if (letsgo->box.Overlap(smartBox))
		{
			smartBox.OverlapWithResponse(letsgo->box, forward);
			return smartBoxCollide(forward, depth + 1);
		}
	}
	return false;
}

void idle()
{
	static int lastFrame = 0;
	counter++;
	const int now = glutGet(GLUT_ELAPSED_TIME);
	const int elapsed = now - lastFrame;

	OBB goober2(AABB(glm::vec3(0), glm::vec3(1)));
	goober2.Translate(glm::vec3(2, 0.1, 0));	
	goober2.Rotate(glm::radians(glm::vec3(0, counter * 4.f, 0)));
	glm::mat4 tester = (goober2.GetModel().GetNormalMatrix());
	//std::cout << "\r" << "AABB Axis: " << goober2.Forward() << "\t Euler Axis" << tester * glm::vec4(1, 0, 0, 0) << std::endl;
	//std::cout << "\r" << "AABB Axis: " << goober2.Forward() << "\t Euler Axis" << glm::transpose(tester)[0];
	//std::cout << "\r" << (float)elapsed / 1000.f << "\t" << smartBox.GetModel().translation;
	


	float speed = 3 * ((float) elapsed) / 1000.f;

	glm::vec3 forward = glm::eulerAngleY(glm::radians(-angles.y)) * glm::vec4(1, 0, 0, 0);
	glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
	forward = speed * glm::normalize(forward);
	right = speed * glm::normalize(right);
	glm::vec3 previous = offset;
	if (keyState[ArrowKeyUp])    smartBox.Translate(smartBox.Forward() * speed);
	if (keyState[ArrowKeyDown])  smartBox.Translate(smartBox.Forward() * -speed);
	if (keyState[ArrowKeyRight]) smartBox.Rotate(glm::vec3(0, -1.f, 0));
	if (keyState[ArrowKeyLeft])  smartBox.Rotate(glm::vec3(0, 1.f, 0));
	if (keyState[ArrowKeyUp] || keyState[ArrowKeyDown] || keyState[ArrowKeyRight] || keyState[ArrowKeyLeft])
	{
		smartBoxColor = false;
		float a = (keyState[ArrowKeyDown] || keyState[ArrowKeyLeft]) ? -1.f : 1.f;
		smartBoxColor = smartBoxCollide(a * smartBox.Forward() * speed);
		/*
		for (auto& wall : boxes)
		{
			smartBoxColor |= smartBox.Overlap(wall.box);
			smartBox.OverlapWithResponse(wall.box, a * smartBox.Forward() * speed);
			//if (smartBoxColor) break;
		}*/
	}
	if (keyState['p'] || keyState['P'])
		std::cout << previous << std::endl;
	if (keyState['w'] || keyState['W'])
		offset += forward;
	if (keyState['s'] || keyState['S'])
		offset -= forward;
	if (keyState['d'] || keyState['D'])
		offset += right;
	if (keyState['a'] || keyState['A'])
		offset -= right;
	if (keyState['k'])
		offset.y = -10;
	if (keyState['b'])
		offset.y = 10;
	if (offset != previous)
	{
		AABB playerBounds(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
		OBB goober(playerBounds);
		playerBounds.Center(offset);
		OBB playerOb(playerBounds);
		//playerOb.Rotate(glm::radians(glm::vec3(0, 45, 0)));

		goober.Translate(glm::vec3(2, 0, 0));
		goober.Rotate(glm::radians(glm::vec3(0, counter * 4.f, 0)));
		for (auto& wall : boxes)
		{
			if (wall.box.Overlap(playerOb))
			{
				offset = previous;
				break;
			}
		}
		if (goober.Overlap(playerOb))
		{
			offset = previous;
		}
		//Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f))
	}
	std::copy(std::begin(keyState), std::end(keyState), std::begin(keyStateBackup));
	std::swap(keyState, keyStateBackup);
	lastFrame = now;
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	keyState[key] = true;
	if (key == 'm' || key == 'M') offset.y += 3;
	if (key == 'n' || key == 'N') offset.y -= 3;
	if (key == 'q' || key == 'Q')
		glutLeaveMainLoop();
	if (key == 't' || key == 'T')
		outlineBoxes = !outlineBoxes;
	if (key == 'g' || key == 'G')
		dummyFlag = !dummyFlag;
	if (key == 'h' || key == 'H')
		clear = !clear;
	if (key == 'r' || key == 'R')
	{
		glm::vec3 angles2 = glm::radians(angles);

		glm::vec3 gamer = glm::normalize(glm::eulerAngleXYZ(-angles2.z, -angles2.y, -angles2.x) * glm::vec4(1, 0, 0, 0));
		std::array<glm::vec3, 8> verts = { offset, offset + gamer * 100.f , offset, offset + gamer * 100.f};
		bool set = false;


		Collision nears, fars;
		//for (std::size_t i = 0; i < boxes.size(); i++)
		for (auto& box: boxes)
		{
			//boxColor[i] = boxes[i].Intersect(offset, gamer * 100.f, nears, fars);
			box.color = box.box.Intersect(offset, gamer * 100.f, nears, fars);
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
		}
		rayBuffer.BufferSubData(verts);
	}
}

void keyboardOff(unsigned char key, int x, int y)
{
	keyState[key] = false;
}

constexpr float ANGLE_DELTA = 4;
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

	angles.x += zDelta;
	angles.y += yDelta;

	previousX = x;
	previousY = y;
}

void specialKeys(int key, [[maybe_unused]] int x, [[maybe_unused]] int y)
{
	switch (key)
	{
	case GLUT_KEY_UP: keyState[ArrowKeyUp] = true; break;
	case GLUT_KEY_DOWN: keyState[ArrowKeyDown] = true; break;
	case GLUT_KEY_RIGHT: keyState[ArrowKeyRight] = true; break;
	case GLUT_KEY_LEFT: keyState[ArrowKeyLeft] = true; break;
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

int main(int argc, char** argv)
{
	int error = 0;

	// Glut
	glutInit(&argc, argv);
	glutInitContextVersion(4, 6);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(1000, 1000);
	glutCreateWindow("Wowie a window");

	glewExperimental = GL_TRUE;
	// Glew
	if ((error = glewInit()) != GLEW_OK)
	{
		printf("Error code %i from glewInit()", error);
		return -1;
	}
	glDisable(GL_MULTISAMPLE);

	Shader::IncludeInShaderFilesystem("FooBarGamer.gsl", "uniformv.glsl");

	uniform.CompileSimple("uniform");
	dither.CompileSimple("light_text_dither");
	CheckError();

	sphereShader.CompileSimple("lightflat");
	CheckError();


	frameShader.CompileSimple("framebuffer");

	texture.Load("text.png");
	wallTexture.Load("flowed.png");
	wallTexture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);

	// Set up VBO/VAO
	glGenVertexArrays(1, &aabbVAO);
	glGenVertexArrays(1, &texturedVAO);

	glGenBuffers(1, &triVBO);
	glGenBuffers(1, &planeBO);
	glGenBuffers(1, &stickBuf);
	glGenBuffers(1, &texturedPlane);
	ColoredVertex data[] = {
		{{-0.5, -0.5, 0}, {1, 0, 0}},
		{{0.5, -0.5, 0}, {0, 1, 0}},
		{{0, 0.5, 0}, {0, 0, 1}}
	};
	glBindBuffer(GL_ARRAY_BUFFER, triVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(ColoredVertex) * 3, data, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, stickBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * stick.size(), stick.data(), GL_STATIC_DRAW);

	CheckError();

	stickVAO.Generate();
	stickVAO.FillArray<Vertex>(uniform);


	glBindBuffer(GL_ARRAY_BUFFER, texturedPlane);
	TextureVertex verts[4] = {};
	for (int i = 0; i < 4; i++)
		verts[i].position = plane[i];
	verts[0].coordinates = glm::vec2(1, 1);
	verts[1].coordinates = glm::vec2(1, 0);
	verts[2].coordinates = glm::vec2(0, 1);
	verts[3].coordinates = glm::vec2(0, 0);
	glBufferData(GL_ARRAY_BUFFER, sizeof(TextureVertex) * 4, verts, GL_STATIC_DRAW);

	CheckError();
	gamerTest.Generate();
	CheckError();
	gamerTest.FillArray<TextureVertex>(dither);
	CheckError();

	glBindBuffer(GL_ARRAY_BUFFER, planeBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 4, plane, GL_STATIC_DRAW);
	CheckError();


	vertexVAO.Generate();
	vertexVAO.FillArray<Vertex>(uniform);

	buffer.Generate(ArrayBuffer);
	buffer.BufferData(plainCubeVerts, StaticDraw);

	buffer.BindBuffer();
	glBindVertexArray(aabbVAO);
	glVertexAttribPointer(uniform.Index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
	glEnableVertexArrayAttrib(aabbVAO, uniform.Index("vPos"));
	glBindVertexArray(0);

	CheckError();

	std::array<glm::vec3, 8> gobs = { glm::vec3(), glm::vec3(5), glm::vec3(3),  glm::vec3(4)};
	rayBuffer.Generate(ArrayBuffer);
	rayBuffer.BufferData(gobs, StaticDraw);
	rayBuffer.BindBuffer();
	rayVAO.Generate();
	rayVAO.FillArray<Vertex>(uniform);

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

	planes.push_back(Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f)));
	for (const auto& ref : planes)
	{
		walls.push_back(Wall(ref));
		OBB project(ref);
		project.Scale(glm::vec3(1, .25f, 1));
		boxes.Insert({project, false}, project.GetAABB());
		boxColor.push_back(false);
	}
	Model oops = planes[planes.size() / 2 + 1];

	ditherTexture.Load(dither16, InternalRed, FormatRed, DataUnsignedByte);
	ditherTexture.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);
	ditherTexture.GenerateMipmap();

	CheckError();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_POLYGON_SMOOTH);

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

	// Framebuffer stuff
	framebufferColor.CreateEmpty(1000, 1000, InternalRGB);
	framebufferColor.SetFilters(MinLinear, MagLinear, Repeat, Repeat);

	framebufferDepth.CreateEmpty(1000, 1000, InternalDepth);
	framebufferDepth.SetFilters(MinLinear, MagLinear, Repeat, Repeat);

	framebufferNormal.CreateEmpty(1000, 1000, InternalRGBA);
	framebufferNormal.SetFilters(MinLinear, MagLinear, Repeat, Repeat);

	normalModifier.CreateEmpty(1000, 1000, InternalRGBA);
	normalModifier.SetFilters(MinLinear, MagLinear, Repeat, Repeat);

	// TODO: Framebuffer class to do this stuff
	// TODO: Renderbuffer for buffers that don't need to be directly read
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferColor.GetGLTexture(), 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, framebufferNormal.GetGLTexture(), 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, framebufferDepth.GetGLTexture(), 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer incomplete ahhhhh" << std::endl;


	glGenFramebuffers(1, &framebufferMod);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferMod);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normalModifier.GetGLTexture(), 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer incomplete ahhhhh" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	// TODO: VAO class
	glGenVertexArrays(1, &frameVAO);
	framebufferBuffer.Generate(ArrayBuffer);
	framebufferBuffer.BufferData(FrameBufferVerts, StaticDraw);
	framebufferBuffer.BindBuffer();
	CheckError();
	glBindVertexArray(frameVAO);
	glVertexAttribPointer(frameShader.Index("positionAndTexture"), 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
	glEnableVertexArrayAttrib(frameVAO, frameShader.Index("positionAndTexture"));
	expand.Compile("framebuffer", "expand");

	auto stuff = GenerateSphere(30, 30);
	sphereBuf = std::get<0>(stuff);
	sphereIndex = std::get<1>(stuff);
	sphereCount = (GLuint) std::get<2>(stuff);

	glBindBuffer(GL_ARRAY_BUFFER, sphereBuf);
	sphereVAO.Generate();
	sphereVAO.FillArray<NormalVertex>(sphereShader);

	hatching.Load("hatching.png");
	hatching.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	uniform.UniformBlockBinding("Camera", 0);
	dither.UniformBlockBinding("Camera", 0);
	sphereShader.UniformBlockBinding("Camera", 0);

	smartBox.Center(glm::vec3(3, 1, 0));
	smartBox.Scale(glm::vec3(0.5f));
	smartBox.Rotate(glm::vec3(0, 90, 0));

	universal.Generate(DynamicDraw, 2 * sizeof(glm::mat4));
	universal.SetBindingPoint(0);
	universal.BindUniform();
	CheckError();
	glm::mat4 projection = glm::perspective(glm::radians(70.f), 1.f, 0.1f, 100.0f);
	universal.BufferSubData(projection, sizeof(glm::mat4));
	CheckError();

	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glClearColor(1.f, 1.f, 1.f, 1.f);
	CheckError();
	glutMainLoop();

	glDeleteFramebuffers(1, &framebuffer);
	return 0;
}