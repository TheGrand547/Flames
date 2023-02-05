#include <iostream>
#include <glew.h>
#include <glm.hpp>
// These are also glm I don't know why they're weird
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>
#include <gtc/type_ptr.hpp>
#include <gtx/euler_angles.hpp>
#include <freeglut.h>
#include "Buffer.h"
#include "Shader.h"
#include "glmHelp.h"
#include "Model.h"
#include "Texture2D.h"
#include "stbWrangler.h"

#define CheckError() CheckErrors(__LINE__);

void CheckErrors(int line)
{
	GLenum e;
	while ((e = glGetError()))
	{
		std::string given((char *) gluErrorString(e));
		std::cout << "Line " << line << ": " << given << std::endl;
	}
}

template <class T> inline void CombineVector(std::vector<T>& left, const std::vector<T>& right)
{
	left.insert(left.end(), std::make_move_iterator(right.begin()), std::make_move_iterator(right.end()));
}

// Cringe globals
GLuint triVBO, planeBO, cubeIndex, vertexVAO, otherVAO;
Shader dammit, other, textures, light, lightTextured;
Buffer buffer;

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

ColoredVertex generic[] =
{
	{{-1, -1, -1}, {1, 0, 0}},
	{{ 1, -1, -1}, {1, 1, 1}},
	{{ 1,  1, -1}, {0, 1, 0}},
	{{-1,  1, -1}, {1, 1, 0}},

	{{-1, -1,  1}, {1, 0, 1}},
	{{ 1, -1,  1}, {1, 1, 0}},
	{{ 1,  1,  1}, {0, 0, 1}},
	{{-1,  1,  1}, {1, 1, 0}},
};

std::array<ColoredVertex, 8> generic2 {
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

// This has one redundant triangle but I can't seem to find it so whatever
// 2,7,6 is repeated in the ord 6, 7, 2 i think i'm not sure ahh
GLubyte index[] =
{
	2, 7, 6, 4, 5, 1, 6, 2, 7, 3, 4, 0, 1, 3, 2
};

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

GLubyte stickDex[] = { 0, 2, 1, 2, 4, 5, 4, 6, 4, 3, 8, 7, 9, 3};
GLuint stickBuf, stickVAO;

GLubyte planeOutline[] = { 0, 1, 3, 2, 0};
Texture2D texture, wallTexture;

static int angle = 0;
static float angleX = 0.f, angleY = 0.f;

glm::vec3 offset(0, 1.5f, 0);
glm::quat rotation(glm::vec3(0, -glm::pi<float>(), 0));
glm::vec3 angles;

GLuint smile, smileVAO;


static const char ditherKernel[] =
{
	0,  32,  8, 40,  2, 34, 10, 42,  
	48, 16, 56, 24, 50, 18, 58, 26, 
	12, 44,  4, 36, 14, 46,  6, 38, 
	60, 28, 52, 20, 62, 30, 54, 22,  
	3,  35, 11, 43,  1, 33,  9, 41, 
	51, 19, 59, 27, 49, 17, 57, 25,
	15, 47,  7, 39, 13, 45,  5, 37,
	63, 31, 55, 23, 61, 29, 53, 21
};

static const unsigned char dither16[16 * 16] = 
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
};

const int ditherSize = 16;

GLuint ditherTexture;
Shader dither;
bool flop = false;
Texture2D perlinTexture;
Shader perlinShader;


enum GeometryThing : unsigned char
{
	PlusX  = 0x01,
	MinusX = 0x02,
	PlusZ  = 0x04,
	MinusZ = 0x08,
	PlusY  = 0x10,
	MinusY = 0x20,
	WallX  = PlusX | MinusX,
	WallZ  = PlusZ | MinusZ,
	All    = 0xFF,
};

std::vector<Model> GetPlaneSegment(const glm::vec3& base, unsigned char flags)
{
	std::vector<Model> results;
	if (flags & PlusX)  results.push_back({ base + glm::vec3(-1, 1,  0), glm::vec3(0, 0, -90.f) });
	if (flags & MinusX) results.push_back({ base + glm::vec3( 1, 1,  0), glm::vec3(0, 0, 90.f) });
	if (flags & PlusZ)  results.push_back({ base + glm::vec3( 0, 1, -1), glm::vec3(90, 0, 0) });
	if (flags & MinusZ) results.push_back({ base + glm::vec3( 0, 1,  1), glm::vec3(-90, 0, 0) });
	if (flags & PlusY)  results.push_back({ base });
	if (flags & MinusY) results.push_back({ base + glm::vec3( 0, 2,  0), glm::vec3(180, 0, 0)});
	return results;
}

std::vector<Model> GetHallway(const glm::vec3& base, bool openZ = true)
{
	return GetPlaneSegment(base, openZ ? PlusX | MinusX | PlusY : PlusZ | MinusZ | PlusY);
}
std::vector<Model> planes;

void display()
{
	// FORWARD IS (0, 0, 1)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	dammit.SetActive();
	glBindVertexArray(vertexVAO);
	glm::mat4 projection = glm::perspective(glm::radians(70.f), 1.f, 0.1f, 100.0f);
	//glm::mat4 pog = RotateY(RotationX(glm::radians(angleX)), glm::radians(angleY));
	glm::mat4 pog = glm::mat4(1.0f);

	//glm::vec3 normal = ((glm::mat4) rotation) * glm::vec3(1, 0, 0);
	glm::vec3 normal = rotation * glm::vec3(1, 0, 0);

	// Camera matrix
	glm::vec3 angles2 = glm::radians(angles);//glm::eulerAngles(rotation);

	//glm::mat4 view = glm::lookAt(offset, offset + Vec4to3(normal), glm::vec3(0, 1, 0));
	//glm::mat4 view = glm::translate((glm::mat4)rotation, -offset);
	// Adding pi is necessary because the default camera is facing -z
	glm::mat4 view = glm::translate(glm::eulerAngleXYZ(angles2.x, angles2.y + glm::pi<float>(), angles2.z), -offset);

	// MVP = model

	pog = projection * view * pog;
	
	if (flop)
	{
		perlinShader.SetActive();
		perlinTexture.Bind(1);
	}
	else
	{
		dither.SetActive();
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, ditherTexture);
	}
	wallTexture.Bind(0);

	glBindBuffer(GL_ARRAY_BUFFER, smile);
	glm::vec3 colors(.5f, .5f, .5f);
	lightTextured.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
	lightTextured.SetVec3("lightPos", glm::vec3(0.f, 1.5f, 0.f));
	lightTextured.SetVec3("viewPos", offset);
	lightTextured.SetMat4("vp", pog);
	lightTextured.SetTextureUnit("textureIn", 0);
	lightTextured.SetTextureUnit("ditherMap", 1);
	//glBindBuffer(GL_ARRAY_BUFFER, planeBO);
	glBindVertexArray(smileVAO);

	for (Model& model : planes)
	{
		glm::vec3 color(.5f, .5f, .5f);
		lightTextured.SetMat4("model", model.GetModelMatrix());
		lightTextured.SetVec3("color", color);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		color = glm::vec3(1, 1, 1);
		lightTextured.SetVec3("color", color);
		glDrawElements(GL_LINE_STRIP, sizeof(planeOutline), GL_UNSIGNED_BYTE, planeOutline);
	}

	dammit.SetActive();
	glBindBuffer(GL_ARRAY_BUFFER, stickBuf);
	glBindVertexArray(stickVAO);
	colors = glm::vec3(1, 0, 0);
	Model m22(glm::vec3(0, 0, 10), glm::vec3(0, 90.f, 0));
	dammit.SetMat4("mvp", pog * m22.GetModelMatrix());
	dammit.SetVec3("color", colors);
	glDrawElements(GL_LINE_STRIP, sizeof(stickDex), GL_UNSIGNED_BYTE, stickDex);

	/*
	other.SetActive();
	buffer.BindBuffer();
	pog = projection * view * RotateY(RotationX(glm::radians(angleX)), glm::radians(angleY)) * 
		glm::scale(glm::mat4(1.f), glm::vec3(0.2, 0.2, 0.2)) * glm::translate(glm::mat4(1.f), glm::vec3(0, .4, 0));
	glUniformMatrix4fv(other.uniformIndex("mvp"), 1, GL_FALSE, glm::value_ptr(pog));
	glBindVertexArray(otherVAO);
	//glDrawElements(GL_TRIANGLE_STRIP, sizeof(index), GL_UNSIGNED_BYTE, index);
	CheckError();

	// I don't know what the hell I'm doing
	glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(0, -1, 0));

	glm::mat4 s = glm::scale(glm::mat4(1.0f), glm::vec3(1, 1.f / 2.f, 5));
	glm::mat4 r = RotateY(RotationX(glm::radians(angleX)), glm::radians(angleY));

	pog = projection * view * (s * r * t);
	//glUniformMatrix4fv(dammit.uniformIndex("rotation"), 1, GL_FALSE, glm::value_ptr(pog));
	//glDrawElements(GL_TRIANGLE_STRIP, 14, GL_UNSIGNED_BYTE, index);
	*/
	glutSwapBuffers();
}

bool spin = false;
std::vector<bool> keyState(UCHAR_MAX);

void idle()
{
	static int lastFrame = 0;
	const int now = glutGet(GLUT_ELAPSED_TIME);
	const int elapsed = now - lastFrame;

	if (spin)
		angle += 1;

	glm::vec3 yRotation = glm::eulerAngles(rotation);
	yRotation.x = yRotation.z = 0;

	float speed = 3 * ((float) elapsed) / 1000.f;

	//glm::vec3 forward = (glm::vec3)(((glm::mat4)glm::quat(yRotation)) * glm::vec4(1, 0, 0, 0));
	glm::vec3 forward = glm::eulerAngleY(glm::radians(-angles.y)) * glm::vec4(0, 0, 1, 0);
	glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
	forward = speed * glm::normalize(forward);
	right   = speed * glm::normalize(right);
	if (keyState['w'] || keyState['W'])
		offset += forward;
	if (keyState['s'] || keyState['S'])
		offset -= forward;
	if (keyState['d'] || keyState['D'])
		offset += right;
	if (keyState['a'] || keyState['A'])
		offset -= right;

	lastFrame = now;
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	keyState[key] = true;
	if (key == 'q' || key == 'Q')
		glutLeaveMainLoop();
	if (key == 'f')
	{
		std::cout << offset.x << ", " << offset.y << ", " << offset.z << std::endl;
		offset = glm::vec3(0, 1.5f, 0);
	}
	if (key == 'g')
	{
		std::cout << "Switched!";
		flop = !flop;
		if (flop)
			std::cout << "to perlin!";
		std::cout << std::endl;
	}
	/*
	glm::vec3 yRotation = glm::eulerAngles(rotation);
	yRotation.x = yRotation.z = 0;
	glm::vec3 forward = (glm::vec3) (((glm::mat4) glm::quat(yRotation)) * glm::vec4(1, 0, 0, 0));
	glm::vec3 right = glm::cross(forward, glm::vec3(0, 1, 0));
	switch (key)
	{
	case 'q': case 'Q':
		glutLeaveMainLoop();
		break;
	case 's': case 'S':
		offset -= forward; break;
	case 'w': case 'W':
		offset += forward; break;
	case 'a': case 'A':
		offset -= right; break;
	case 'd': case 'D':
		offset += right; break;
	}*/
}

void keyboardOff(unsigned char key, int x, int y)
{
	keyState[key] = false;
}

#define ANGLE_DELTA 4
void mouseFunc(int x, int y)
{
	static int previousX, previousY;
	float xDif = (float) x - previousX;
	float yDif = (float) y - previousY;
	if (abs(xDif) > 20)
		xDif = 0;
	if (abs(yDif) > 20)
		yDif = 0;
	float yDelta = 50 * (xDif * ANGLE_DELTA) / glutGet(GLUT_WINDOW_WIDTH);
	float zDelta = 50 * (yDif * ANGLE_DELTA) / glutGet(GLUT_WINDOW_HEIGHT);
	glm::quat delta(glm::radians(glm::vec3(zDelta, yDelta, 0)));

	angles.x += zDelta;
	angles.y += yDelta;

	//glm::quat delta(glm::radians(glm::vec3(0, yDelta, zDelta) / 1.f));
	//glm::quat delta(glm::radians(glm::vec3(0, yDelta, zDelta)));
	//rotation = (-delta) * rotation * delta;
	//glm::eua
	rotation = rotation * delta;
	//glm::vec3 goop = glm::eulerAngles(rotation);
	/*
	goop.x = 0;
	rotation = glm::quat(goop);*/
	previousX = x;
	previousY = y;
}

int main(int argc, char** argv)
{
	int error = 0;

	// Glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(512, 512);
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
	perlinShader.Compile("light_text_dither", "light_text_perlin");

	other.CompileSimple("test");

	textures.CompileSimple("texture");

	texture.Load("test.png");
	wallTexture.Load("wall2.png");

	// Set up VBO/VAO
	glGenVertexArrays(1, &vertexVAO);
	glGenVertexArrays(1, &otherVAO);
	glGenVertexArrays(1, &stickVAO);
	glGenVertexArrays(1, &smileVAO);

	glGenBuffers(1, &triVBO);
	glGenBuffers(1, &planeBO);
	glGenBuffers(1, &stickBuf);
	glGenBuffers(1, &smile);
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


	glBindBuffer(GL_ARRAY_BUFFER, smile);
	TextureVertex verts[4] = {};
	for (int i = 0; i < 4; i++)
		verts[i].position = plane[i];
	verts[0].coordinates = glm::vec2(1, 1);
	verts[1].coordinates = glm::vec2(1, 0);
	verts[2].coordinates = glm::vec2(0, 1);
	verts[3].coordinates = glm::vec2(0, 0);
	glBufferData(GL_ARRAY_BUFFER, sizeof(TextureVertex) * 4, verts, GL_STATIC_DRAW);

	glBindVertexArray(smileVAO);
	glVertexAttribPointer(textures.index("pos"), 3, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), nullptr);
	glVertexAttribPointer(textures.index("tex"), 2, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (const void*) offsetof(TextureVertex, coordinates));
	glEnableVertexArrayAttrib(smileVAO, textures.index("pos"));
	glEnableVertexArrayAttrib(smileVAO, textures.index("tex"));


	glBindBuffer(GL_ARRAY_BUFFER, planeBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 4, plane, GL_STATIC_DRAW);

	glBindVertexArray(vertexVAO);
	glVertexAttribPointer(dammit.index("pos"), 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
	glEnableVertexArrayAttrib(vertexVAO, dammit.index("pos"));


	buffer.Generate(GL_ARRAY_BUFFER);
	buffer.BufferData(generic2, GL_STATIC_DRAW);
	buffer.BindBuffer();
	glBindVertexArray(otherVAO);
	glVertexAttribPointer(other.index("pos"), 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), nullptr);
	glEnableVertexArrayAttrib(otherVAO, other.index("pos"));
	
	glVertexAttribPointer(other.index("color"), 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), (const void*) offsetof(ColoredVertex, color));
	glEnableVertexArrayAttrib(otherVAO, other.index("color"));

	glm::vec3 origin(0, 0, 0);
	CombineVector(planes, GetPlaneSegment(origin, PlusY));
	for (int i = -5; i <= 5; i++)
	{
		if (!i)
			continue;
		CombineVector(planes, GetHallway(glm::vec3(0, 0, 2 * i), true));
		CombineVector(planes, GetHallway(glm::vec3(2 * i, 0, 0), false));
	}
	


	glGenTextures(1, &ditherTexture);
	glBindTexture(GL_TEXTURE_2D, ditherTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, 1, 16, 16, 0, GL_RED, GL_UNSIGNED_BYTE, dither16);
	glGenerateMipmap(GL_TEXTURE_2D);

	CheckError();
	// TODO: Add perlin noise thingy
	{
		std::array<float, 160 * 160> perlinArray{};
		for (std::size_t i = 0; i < 160; i++)
		{
			for (std::size_t j = 0; j < 160; j++)
			{
				perlinArray[i * 160 + j] = stb_perlin_noise3(i * 0.352f, j * 0.432f, (i + j) * 0.7945f, 0, 0, 0) / 2.0f + 0.5f;
			}
		}
		perlinTexture.Load(perlinArray);
	}


	CheckError();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// Not indexed color, irrelevant
	//glEnable(GL_DITHER);

	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_POLYGON_SMOOTH);


	glDepthFunc(GL_LESS);
	glClearColor(0, 0, 0, 1);
	glLineWidth(5);

	glutDisplayFunc(display);
	glutIdleFunc(idle);

	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardOff);

	glutMotionFunc(mouseFunc);
	glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);

	glutMainLoop();
	return 0;
}