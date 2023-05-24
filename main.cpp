#include <iostream>
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <freeglut.h>
#include "AABB.h"
#include "Buffer.h"
#include "Shader.h"
#include "glmHelp.h"
#include "Model.h"
#include "OrientedBoundingBox.h"
#include "Texture2D.h"
#include "stbWrangler.h"
#include "Plane.h"
#include "Sphere.h"
#include "Wall.h"

#define CheckError() CheckErrors(__LINE__);

void CheckErrors(int line)
{
	GLenum e;
	while ((e = glGetError()))
	{
		std::string given((char*)gluErrorString(e));
		std::cout << "Line " << line << ": " << given << std::endl;
	}
}

template <class T> inline void CombineVector(std::vector<T>& left, const std::vector<T>& right)
{
	left.insert(left.end(), std::make_move_iterator(right.begin()), std::make_move_iterator(right.end()));
}

// Cringe globals
GLuint triVBO, planeBO, cubeIndex, vertexVAO, aabbVAO;
Shader dammit, aabbShader, textures, light, lightTextured;
Buffer buffer;

GLuint sphereBuf, sphereIndex, sphereVAO, sphereCount;
Shader sphereShader;

struct ColoredVertex
{
	glm::vec3 position;
	glm::vec3 color;
};

struct TextureVertex
{
	glm::vec3 position;
	glm::vec2 coordinates;
};

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

std::array<glm::vec3, 8> plainCubeVerts{
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

// This has one redundant triangle but I can't seem to find it so whatever
// 2,7,6 is repeated in the ord 6, 7, 2 i think i'm not sure ahh
std::array<GLubyte, 36> cubeIndicies =
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

/*
GLubyte index2[] =
{
	//2, 3, 1, 0, 4, 5, 6, 7, 8, 6, 4, 3, 2, 1
	0, 3, 1, 2, 5, 6, 4, 7, 3, 6, 2
};*/

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
GLuint stickBuf, stickVAO;

GLubyte planeOutline[] = { 0, 1, 3, 2, 0 };
Texture2D texture, wallTexture;

bool outlineBoxes = false;

glm::vec3 offset(0, 1.5f, 0);
glm::vec3 angles;

GLuint texturedPlane, texturedVAO;

static const std::array<const unsigned char, 16 * 16> dither16 = {
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


enum GeometryThing : unsigned char
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
std::vector<OBB> boxes;

bool dummyFlag = false;
bool clear = false;
void display()
{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	GLenum buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, buffers);
	CheckError();
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// FORWARD IS (1, 0, 0)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	dammit.SetActive();
	glBindVertexArray(vertexVAO);
	glm::mat4 projection = glm::perspective(glm::radians(70.f), 1.f, 0.1f, 100.0f);
	glm::mat4 projectionView = glm::mat4(1.0f);

	// Camera matrix
	glm::vec3 angles2 = glm::radians(angles);//glm::eulerAngles(rotation);

	// Adding pi/2 is necessary because the default camera is facing -z
	glm::mat4 view = glm::translate(glm::eulerAngleXYZ(angles2.x, angles2.y + glm::half_pi<float>(), angles2.z), -offset);

	// MVP = model

	projectionView = projection * view;


	dither.SetActive();
	wallTexture.Bind(0);
	ditherTexture.Bind(1);

	glBindBuffer(GL_ARRAY_BUFFER, texturedPlane);
	glm::vec3 colors(.5f, .5f, .5f);
	dither.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	dither.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	dither.SetVec3("viewPos", offset);
	dither.SetMat4("vp", projectionView);
	dither.SetTextureUnit("textureIn", 0);
	dither.SetTextureUnit("ditherMap", 1);
	glBindVertexArray(texturedVAO);
	
	for (Model& model : planes)
	{
		glm::vec3 color(.5f, .5f, .5f);
		dither.SetMat4("model", model.GetModelMatrix());
		dither.SetVec3("color", color);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}
	dammit.SetActive();
	glBindBuffer(GL_ARRAY_BUFFER, stickBuf);
	glBindVertexArray(stickVAO);
	colors = glm::vec3(1, 0, 0);
	Model m22(glm::vec3(10, 0, 0));
	dammit.SetMat4("mvp", projectionView * m22.GetModelMatrix());
	dammit.SetVec3("color", colors);
	glDrawElements(GL_LINE_STRIP, sizeof(stickDex), GL_UNSIGNED_BYTE, stickDex);

	if (outlineBoxes)
	{
		aabbShader.SetActive();
		aabbShader.SetVec3("color", colors);
		glm::vec3 blue(0, 0, 1);
		glBindVertexArray(aabbVAO);

		if (dummyFlag)
			glDisable(GL_CULL_FACE);
		
		for (const auto& box : boxes)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			aabbShader.SetVec3("color", colors);
			glm::mat4 boxMat = projectionView * box.GetModel().GetModelMatrix();
			aabbShader.SetMat4("mvp", boxMat);
			glDrawElements(GL_TRIANGLES, cubeIndicies.size(), GL_UNSIGNED_BYTE, cubeIndicies.data());
			aabbShader.SetVec3("color", blue);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawElements(GL_TRIANGLES, cubeIndicies.size(), GL_UNSIGNED_BYTE, cubeIndicies.data());
		}
		glEnable(GL_CULL_FACE);
	}

	glEnable(GL_CULL_FACE);
	sphereShader.SetActive();
	glBindVertexArray(sphereVAO);
	Model sphereModel(glm::vec3(6.5f, 1.5f, 0.f));
	sphereModel.scale = glm::vec3(0.5f);


	hatching.Bind(0);
	sphereShader.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	sphereShader.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
	sphereShader.SetVec3("viewPos", offset);
	sphereShader.SetVec3("shapeColor", glm::vec3(1.f, .75f, 0.f));
	sphereShader.SetMat4("modelMat", sphereModel.GetModelMatrix());
	sphereShader.SetMat4("normMat", sphereModel.GetNormalMatrix());
	sphereShader.SetMat4("viewProjMat", projectionView);
	sphereShader.SetTextureUnit("hatching", 0);

	// Doing this while letting the normal be the color will create a cool effect
	//glDrawArrays(GL_TRIANGLES, 0, 1836);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIndex);
	// Calling with triangle_strip is fucky
	glDrawElements(GL_TRIANGLES, sphereCount, GL_UNSIGNED_INT, nullptr);



	// Framebuffer stuff
	CheckError();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferMod);
	glDrawBuffers(1, buffers);
	glClear(GL_COLOR_BUFFER_BIT);
	
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	framebufferNormal.Bind(0);
	//framebufferDepth.Bind(0);

	frameShader.SetActive();
	frameShader.SetTextureUnit("normal", 0);
	glBindVertexArray(frameVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	framebufferColor.Bind(0);
	normalModifier.Bind(1);
	framebufferDepth.Bind(2);
	expand.SetActive();
	expand.SetTextureUnit("screen", 0);
	expand.SetTextureUnit("edges", 1);
	expand.SetTextureUnit("depths", 2);
	expand.SetInt("depth", 1);
	glBindVertexArray(frameVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);


	glutSwapBuffers();
	CheckError();
}

std::vector<bool> keyState(UCHAR_MAX);
std::vector<Wall> walls;

// To get a perpendicular vector to a vector <a, b, c> do that cross <1, 0, 0> to get <0, c, -b>

void idle()
{
	static int lastFrame = 0;
	const int now = glutGet(GLUT_ELAPSED_TIME);
	const int elapsed = now - lastFrame;

	float speed = 3 * ((float)elapsed) / 1000.f;

	glm::vec3 forward = glm::eulerAngleY(glm::radians(-angles.y)) * glm::vec4(1, 0, 0, 0);
	glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
	forward = speed * glm::normalize(forward);
	right = speed * glm::normalize(right);
	glm::vec3 previous = offset;
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
		playerBounds.Center(offset);

		for (auto& wall : boxes)
		{
			if (wall.Overlap(OBB(playerBounds)))
			{
				offset = previous;
				break;
			}
		}
	}

	lastFrame = now;
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	keyState[key] = true;
	if (key == 'q' || key == 'Q')
		glutLeaveMainLoop();
	if (key == 't' || key == 'T')
		outlineBoxes = !outlineBoxes;
	if (key == 'g' || key == 'G')
		dummyFlag = !dummyFlag;
	if (key == 'h' || key == 'H')
		clear = !clear;
	if (key == 'f')
	{
		std::cout << offset.x << ", " << offset.y << ", " << offset.z << std::endl;
		offset = glm::vec3(1, 1.5f, 1);
	}
}

void keyboardOff(unsigned char key, int x, int y)
{
	keyState[key] = false;
}

#define ANGLE_DELTA 4
void mouseFunc(int x, int y)
{
	static int previousX, previousY;
	float xDif = (float)x - previousX;
	float yDif = (float)y - previousY;
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

int main(int argc, char** argv)
{
	int error = 0;

	// Glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(1000, 1000);
	glutCreateWindow("Wowie a window");

	// Glew
	if ((error = glewInit()) != GLEW_OK)
	{
		printf("Error code %i from glewInit()", error);
		return -1;
	}

	dammit.CompileSimple("uniform");
	light.CompileSimple("light");
	lightTextured.CompileSimple("lighttex");
	dither.CompileSimple("light_text_dither");

	aabbShader.CompileSimple("uniform");
	frameShader.CompileSimple("framebuffer");

	textures.CompileSimple("texture");

	sphereShader.CompileSimple("lightflat");

	texture.Load("test.png");
	wallTexture.Load("flowed.png");
	wallTexture.SetMinFilter(LinearLinear);
	wallTexture.SetMagFilter(MagNearest);

	// Set up VBO/VAO
	glGenVertexArrays(1, &vertexVAO);
	glGenVertexArrays(1, &aabbVAO);
	glGenVertexArrays(1, &stickVAO);
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
	glBindVertexArray(stickVAO);
	glVertexAttribPointer(dammit.index("pos"), 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
	glEnableVertexArrayAttrib(stickVAO, dammit.index("pos"));


	glBindBuffer(GL_ARRAY_BUFFER, texturedPlane);
	TextureVertex verts[4] = {};
	for (int i = 0; i < 4; i++)
		verts[i].position = plane[i];
	verts[0].coordinates = glm::vec2(1, 1);
	verts[1].coordinates = glm::vec2(1, 0);
	verts[2].coordinates = glm::vec2(0, 1);
	verts[3].coordinates = glm::vec2(0, 0);
	glBufferData(GL_ARRAY_BUFFER, sizeof(TextureVertex) * 4, verts, GL_STATIC_DRAW);

	glBindVertexArray(texturedVAO);
	glVertexAttribPointer(textures.index("pos"), 3, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (const void*) nullptr);
	glVertexAttribPointer(textures.index("tex"), 2, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (const void*) offsetof(TextureVertex, coordinates));
	glEnableVertexArrayAttrib(texturedVAO, textures.index("pos"));
	glEnableVertexArrayAttrib(texturedVAO, textures.index("tex"));


	glBindBuffer(GL_ARRAY_BUFFER, planeBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 4, plane, GL_STATIC_DRAW);

	glBindVertexArray(vertexVAO);
	glVertexAttribPointer(dammit.index("pos"), 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
	glEnableVertexArrayAttrib(vertexVAO, dammit.index("pos"));


	buffer.Generate(GL_ARRAY_BUFFER);
	buffer.BufferData(plainCubeVerts, GL_STATIC_DRAW);
	buffer.BindBuffer();
	glBindVertexArray(aabbVAO);
	glVertexAttribPointer(aabbShader.index("pos"), 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
	glEnableVertexArrayAttrib(aabbVAO, aabbShader.index("pos"));


	glm::vec3 origin(0, 0, 0);
	CombineVector(planes, GetPlaneSegment(origin, PlusY));
	for (int i = -5; i <= 5; i++)
	{
		if (!i)
			continue;
		CombineVector(planes, GetHallway(glm::vec3(0, 0, 2 * i), true));
		CombineVector(planes, GetHallway(glm::vec3(2 * i, 0, 0), false));
	}
	planes.push_back(Model(glm::vec3(-3.f, 1.5f, 0), glm::vec3(-23.f, 0, -45.f)));
	for (const auto& ref : planes)
	{
		walls.push_back(Wall(ref));
	}
	Model oops = planes[planes.size() / 2 + 1];
	
	boxes.push_back(AABB(glm::vec3(1, 0, 1), glm::vec3(10, 2, 10)));
	boxes.push_back(AABB::GetAABB(std::vector<glm::vec3>{glm::vec3(-1, 0, -1), glm::vec3(-10, 2, -10)}));
	boxes.push_back(AABB::GetAABB(std::vector<glm::vec3>{glm::vec3(-1, 0,  1), glm::vec3(-10, 2,  10)}));
	boxes.push_back(AABB::GetAABB(std::vector<glm::vec3>{glm::vec3( 1, 0, -1), glm::vec3( 10, 2, -10)}));

	ditherTexture.Load(dither16);
	ditherTexture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);
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

	glutMotionFunc(mouseFunc);
	glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);

	// Framebuffer stuff
	framebufferColor.CreateEmpty(1000, 1000, GL_RGB);
	framebufferColor.SetFilters(MinLinear, MagLinear, Repeat, Repeat);
	framebufferDepth.CreateEmpty(1000, 1000, GL_DEPTH_COMPONENT);
	framebufferDepth.SetFilters(MinLinear, MagLinear, Repeat, Repeat);
	framebufferNormal.CreateEmpty(1000, 1000, GL_RGBA);
	framebufferNormal.SetFilters(MinLinear, MagLinear, Repeat, Repeat);

	normalModifier.CreateEmpty(1000, 1000, GL_RGBA);
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
	framebufferBuffer.Generate(GL_ARRAY_BUFFER);
	framebufferBuffer.BufferData(FrameBufferVerts, GL_STATIC_DRAW);
	framebufferBuffer.BindBuffer();
	CheckError();
	glBindVertexArray(frameVAO);
	glVertexAttribPointer(frameShader.index("positionAndTexture"), 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
	glEnableVertexArrayAttrib(frameVAO, frameShader.index("positionAndTexture"));

	expand.Compile("framebuffer", "expand");

	auto stuff = GenerateSphere(30, 30);
	sphereBuf = std::get<0>(stuff);
	sphereIndex = std::get<1>(stuff);
	sphereCount = (GLuint) std::get<2>(stuff);

	glBindBuffer(GL_ARRAY_BUFFER, sphereBuf);
	glGenVertexArrays(1, &sphereVAO);
	glBindVertexArray(sphereVAO);
	glVertexAttribPointer(sphereShader.index("vPos"), 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), nullptr);
	glEnableVertexArrayAttrib(sphereVAO, sphereShader.index("vPos"));
	glVertexAttribPointer(sphereShader.index("vNorm"), 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (const void*) sizeof(glm::vec3));
	glEnableVertexArrayAttrib(sphereVAO, sphereShader.index("vNorm"));

	hatching.Load("hatching.png");
	hatching.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	int ins;	
	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &ins);

	std::cout << ins << std::endl;

	glClearColor(1.f, 1.f, 1.f, 1.f);

	glutMainLoop();

	glDeleteFramebuffers(1, &framebuffer);
	return 0;
}