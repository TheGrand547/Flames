#include <algorithm>
#include <chrono>
#include <execution>
#include <glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/ulp.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/orthonormalize.hpp>
#include <glm/gtx/vec_swizzle.hpp>
#include <glm/gtx/color_space.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <sys/utime.h>
#include <time.h>
#include <unordered_map>
#include "AABB.h"
#include "Animation.h"
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
#include "DemoGuy.h"
#include "kdTree.h"
#include "Level.h"
#include "Geometry.h"
#include "ExhaustManager.h"
#include "Player.h"
#include "TimeAverage.h"
#include "Satelite.h"
#include "Parallel.h"
#include "DebrisManager.h"
#include "MissileMotion.h"
#include "MagneticAttack.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_impl_glfw.h"
#include "ClockBrain.h"
#include "ShipManager.h"
#include "async/BufferSync.h"
#include "ResourceBank.h"
#include "NavMesh.h"
#include "BinarySpacePartition.h"
#include "Audio.h"
#include "Frustum.h"
#include "DummyArrays.h"
#include "Door.h"
#include "entities/ShieldGenerator.h"
#include "misc/ExternalShaders.h"
#include <semaphore>

// TODO: https://github.com/zeux/meshoptimizer once you use meshes
// TODO: Delaunay Trianglulation
// TODO: EASTL

// Stencil based limited vision range
// RTWP first person vaguely rpg
// Most actions take time beyond just the input, not (just) a delay
// Can be buffered
// UI element showing distance to cursor at all times to better judge movement
// 128 gameplay ticks a second allowing for 1/16th speed play(wowie slow motion)
// Animations and such are locked to this "grid"
// TODO: Stencil buffer for vision cones and things

ASCIIFont fonter;

// Buffers
ArrayBuffer albertBuffer, textBuffer, capsuleBuffer, rayBuffer, sphereBuffer, stickBuffer;
ArrayBuffer decals;
ArrayBuffer exhaustBuffer;
ArrayBuffer leftBuffer, rightBuffer;

MeshData guyMeshData;

ElementArray capsuleIndex, cubeOutlineIndex, solidCubeIndex, sphereIndicies, stickIndicies;

UniformBuffer cameraUniformBuffer, pointUniformBuffer, screenSpaceBuffer;

// TODO: TODO: TODO: make it not vec4's and use not std140 layout
UniformBuffer lightingBuffer;

// Shaders
Shader fontShader, uiRect, uiRectTexture, uniform, widget;
Shader decalShader;
Shader pathNodeView, stencilTest;
Shader nineSlicer;
Shader billboardShader;
Shader vision;
Shader ship;
Shader basic;
Shader engine;
Shader skyBox;
Shader trails;
Shader debris;

// Textures
Texture2D ditherTexture, hatching, normalMap, texture, wallTexture;
Texture2D buttonA, buttonB, nineSlice;
CubeMap sky;

// Vertex Array Objects
VAO fontVAO, pathNodeVAO, meshVAO, plainVAO, texturedVAO;
VAO nineSliced, billboardVAO;
VAO colorVAO;

// Not explicitly tied to OpenGL Globals

OBB dumbBox; // rip smartbox
static unsigned int idleFrameCounter = 0;

constexpr auto TIGHT_BOXES = 2;
constexpr auto FREEZE_GAMEPLAY = 1;
constexpr auto DEBUG_PATH = 4;
constexpr auto DYNAMIC_TREE = 5;
constexpr auto FULL_CALCULATIONS = 5;
constexpr auto CHECK_UVS = 3;
// One for each number key
std::array<bool, '9' - '0' + 1> debugFlags{};

// Input Shenanigans
constexpr auto ArrowKeyUp = 0;
constexpr auto ArrowKeyDown = 1;
constexpr auto ArrowKeyRight = 2;
constexpr auto ArrowKeyLeft = 3;

std::array<bool, UCHAR_MAX> keyState{}, keyStateBackup{};

// TODO: Breaking people out of an prison station that had been abandoned due to upcoming supernova or something
// Only automated guards remain and you have three characters you switch between at will, which also act as your lives
// Guard manipulation is a core aspect, they are unkillable but can be disabled before being turned back on by smaller things
// Vaguely turn based, you decide what your guy does for the next [5,10,15] seconds then it plays out
// Enemy behavior is entirely predictable for the duration of a turn
// Sonar scanning of environment(imperfect information) required beyond a very limited vision range


ColorFrameBuffer playerTextEntry;
std::stringstream letters("abc");
bool reRenderText = true;

constexpr float ANGLE_DELTA = 4;

// Camera
glm::vec3 cameraPosition(0, 1.5f, 0);
glm::vec3 cameraRotation(0, 0, 0);

float zNear = 0.1f, zFar = 1000.f;

bool buttonToggle = false;
ScreenRect buttonRect{ 540, 200, 100, 100 }, userPortion(0, 800, 1000, 200);
Button help(buttonRect, [](std::size_t i) {std::cout << idleFrameCounter << std::endl; });

bool featureToggle = false;
std::chrono::nanoseconds idleTime, displayTime, renderDelay;

struct LightVolume
{
	glm::vec4 position;
	glm::vec3 color;
	glm::vec3 constants;
};

DynamicOctTree<PathFollower> followers{AABB(glm::vec3(1000.f))};

// TODO: Semaphore version of buffersync
BufferSync<std::vector<TextureVertex>> decalVertex;

std::array<ScreenRect, 9> ui_tester;
ArrayBuffer ui_tester_buffer;

ArrayBuffer billboardBuffer, billboardMatrix;

std::vector<AABB> dynamicTreeBoxes;
using namespace Input;

bool shouldClose = false;

using TimePoint = std::chrono::steady_clock::time_point;
using TimeDelta = std::chrono::nanoseconds;
static std::size_t gameTicks = 0;

SimpleAnimation foobar{ {glm::vec3(-0.025, 0, 0)}, 32, Easing::Quintic,
						{glm::vec3(-0.25, 0, 0)}, 80, Easing::Linear };
AnimationInstance foobarInstance;

Animation flubber = make_animation( Transform(),
	{
		{{glm::vec3(), glm::quat(glm::radians(glm::vec3(0.f, 0.f, -180.f)))}, 120, Easing::EaseOutCubic},
		{{glm::vec3(), glm::quat(glm::radians(glm::vec3(0.f, 0.f, 270.f)))}, 120, Easing::EaseOutCubic},
		{{glm::vec3(), glm::quat(glm::radians(glm::vec3(10.f, 0, 32.f)))}, 120, Easing::EaseOutQuadratic},
		{{glm::vec3(), glm::quat(glm::radians(glm::vec3(10.f, 0, 32.f)))}, 120, Easing::EaseOutQuadratic},
		{{glm::vec3(), glm::quat()}, 120, Easing::EaseOutQuartic},
	}
);

ExhaustManager managedProcess;

//Player playfield(glm::vec3(-40.f, 60.f, 0.f));
Player playfield(glm::vec3(0.f, 50.f, 0.f));
float playerSpeedControl = 0.1f;
Input::Keyboard boardState; 
// TODO: Proper start/reset value
glm::quat aboutTheShip(0.f, 0.f, 0.f, 1.f);

Satelite groovy{ glm::vec3(10.f, 10.f, 0) };
bool shiftHeld;
std::atomic_uchar addExplosion;

DebrisManager trashMan;
Shader normalDebris;

MagneticAttack magnetic(100, 20, 80, 4.f);
MeshData playerMesh, playerMesh2;
MeshData bulletMesh;

ArrayBuffer bulletMats;
VAO bulletVAO;

BufferSync<std::vector<glm::mat4>> bulletMatricies, bulletImpacts;

MeshData levelGeometry;
ShipManager management;

ClockBrain tickTockMan;

static GLFWwindow* windowPointer = nullptr;
ArrayBuffer levelOutline;

const float BulletDecalScale = 4.f;

UniformBuffer globalLighting;

Framebuffer<3, Depth> deferredBuffer;

// Could be expanded to have another buffer if necessary
ColorFrameBuffer pointLightBuffer;

glm::vec4 testCameraPos(-30.f, 15.f, 0.f, 60.f);
BufferSync<std::vector<LightVolume>> drawingVolumes;
std::vector<LightVolume> constantLights;

Door heWhoSleeps(glm::vec3(97.244f, 17.102f, 0));

glm::vec3 GetCameraFocus(const Model& playerModel, const glm::vec3& velocity)
{
	return playerModel.translation + (playerModel.rotation * glm::vec3(1.f, 0.f, 0.f)) * (10.f + Rectify(glm::length(velocity)) / 2.f);
}

std::pair<glm::vec3, glm::vec3> CalculateCameraPositionDir(const Model& playerModel)
{
	glm::vec3 localCamera = cameraPosition;
	const glm::vec3 velocity = playfield.GetVelocity();
	glm::vec3 basePoint = glm::vec3(4.f, -2.5f, 0.f);
	
	localCamera = (playerModel.rotation * aboutTheShip) * basePoint;
	localCamera += playerModel.translation;
	localCamera -= velocity / 20.f;
	const glm::vec3 cameraFocus = GetCameraFocus(playerModel, velocity);
	const glm::vec3 cameraForward = glm::normalize(cameraFocus - localCamera);
	return { localCamera, cameraForward };
}

Frustum GetFrustum(const Model& playerModel)
{
	auto cameraPair = CalculateCameraPositionDir(playerModel);
	return Frustum(cameraPair.first, ForwardDir(cameraPair.second, playerModel.rotation * glm::vec3(0.f, 1.f, 0.f)), glm::vec2(zNear, zFar));
}

ShieldGenerator bobert;
ColorFrameBuffer buffet;
BufferSync<std::vector<glm::vec3>> shieldPos;
void display()
{
	// Some kind of framerate limiter?
	/*
	static std::size_t lastRenderTick = 0;
	if (lastRenderTick == gameTicks)
	{
		return;
	}
	lastRenderTick = gameTicks;
	*/

	auto displayStartTime = std::chrono::high_resolution_clock::now();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	Window::Viewport();
	glClearColor(0, 0, 0, 1);

	EnableGLFeatures<DepthTesting | FaceCulling>();
	EnableDepthBufferWrite();
	//glClearDepth(0);
	ClearFramebuffer<ColorBuffer | DepthBuffer | StencilBuffer>();
	glClearDepth(1);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	DisableGLFeatures<StencilTesting>();

	{
		if (buffet.GetColor().GetGLTexture() == 0)
		{
			buffet.GetColor().CreateEmpty(glm::ivec2(256), InternalRed16);
			buffet.GetColor().SetFilters(LinearLinear, MagLinear, Repeat, Repeat);
			buffet.GetColor().GenerateMipmap();
			buffet.Assemble();
			buffet.Bind();
		}
		else
		{
			buffet.Bind();
			ClearFramebuffer<ColorBuffer>();
		}
		FeatureFlagPush<FaceCulling | DepthTesting, false> pushed;
		Shader& local = ShaderBank::Get("ShieldTexture");
		local.SetActiveShader();
		local.SetFloat("FrameTime", gameTicks * Tick::TimeDelta);
		local.DrawArray<DrawType::TriangleStrip>(4);
		BindDefaultFrameBuffer();
	}
	Window::Viewport();
	
	const Model playerModel(playfield.GetModel());

	// Camera matrix
	const glm::mat3 axes(playerModel.rotation);
	const glm::vec3 velocity = playfield.GetVelocity();
	
	std::pair<glm::vec3, glm::vec3> cameraPair = CalculateCameraPositionDir(playerModel);
	const glm::vec3 localCamera = cameraPair.first;
	const glm::vec3 cameraForward = cameraPair.second;

	glm::mat4 view = glm::lookAt(localCamera, GetCameraFocus(playerModel, velocity), axes[1]);
	cameraUniformBuffer.BufferSubData(view, 0);
	Frustum frustum(localCamera, ForwardDir(cameraForward, axes[1]), glm::vec2(zNear, zFar));
	CheckError();
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	
	if (debugFlags[DEBUG_PATH])
	{
		EnableGLFeatures<Blending>();
		//DisableDepthBufferWrite();
		pathNodeView.SetActiveShader();
		pathNodeVAO.Bind();
		pathNodeVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"), 0);
		pathNodeVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("nodePositions"), 1);
		pathNodeView.SetFloat("Scale", (glm::cos(idleFrameCounter / 200.f) * 0.05f) + 0.3f);
		pathNodeView.SetVec4("Color", glm::vec4(0, 0, 1, 0.75f));
		
		pathNodeView.DrawElementsInstanced<DrawType::Triangle>(solidCubeIndex, Bank<ArrayBuffer>::Get("nodeLinePositions"));

		uniform.SetActiveShader();
		uniform.SetMat4("Model", glm::mat4(1.f));
		plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("nodeLinePositions"));
		glLineWidth(10.f);
		uniform.DrawArray<DrawType::Lines>(Bank<ArrayBuffer>::Get("nodeLinePositions"));
		uniform.DrawArray<DrawType::LineStrip>(Bank<ArrayBuffer>::Get("nodeLinePositions"));
		//plainVAO.BindArrayBuffer(pathNodePositions);
		//uniform.DrawArray<DrawType::LineStrip>(pathNodePositions);
		//DisableGLFeatures<StencilTesting>();
		EnableDepthBufferWrite();
	}

	// Switch between forward and deferred rendering
	if (debugFlags[FULL_CALCULATIONS] && false)
	{
		Shader& interzone = ShaderBank::Get("defer");
		deferredBuffer.Bind();
		ClearFramebuffer<ColorBuffer | DepthBuffer>();
		VAO& outerzone = VAOBank::Get("new_mesh");
		interzone.SetActiveShader();
		levelGeometry.Bind(outerzone);
		outerzone.BindArrayBuffer(levelGeometry.vertex, 0);
		outerzone.BindArrayBuffer(Bank<ArrayBuffer>::Get("dummyInstance"), 1);
		interzone.SetVec3("shapeColor", glm::vec3(1.0, 1.0, 1.0));
		interzone.DrawElements<DrawType::Triangle>(levelGeometry.indirect);

		auto& buf = BufferBank::Get("player");
		auto meshs2 = playerModel;
		meshs2.scale *= 0.5f;
		auto meshs = meshs2.GetMatrixPair();
		buf.BufferData(std::to_array({ meshs.model, meshs.normal }));
		outerzone.BindArrayBuffer(buf, 1);
		playfield.Draw(interzone, outerzone, playerMesh2, playerModel);

		// Do the actual deferred rendering
		BindDefaultFrameBuffer();
		Shader& interzone2 = ShaderBank::Get("fullRender");
		interzone2.SetActiveShader();
		interzone2.SetInt("featureToggle", featureToggle);
		interzone2.SetVec3("lightPos", playerModel.translation);
		interzone2.SetVec3("lightDir", playerModel.rotation * glm::vec3(1.f, 0.f, 0.f));
		interzone2.SetTextureUnit("position", deferredBuffer.GetColorBuffer<0>(), 0);
		interzone2.SetTextureUnit("normal", deferredBuffer.GetColorBuffer<1>(), 1);
		interzone2.SetTextureUnit("color", deferredBuffer.GetColorBuffer<2>(), 2);
		interzone2.DrawArray<DrawType::TriangleStrip>(4);
	}
	else
	{
		glDepthFunc(GL_LEQUAL);
		//drawingVolumes.ExclusiveOperation([](auto& data) {Bank<ArrayBuffer>::Get("light_volume_mesh").BufferData(data, DynamicDraw); });

		Shader& interzone = ShaderBank::Get("defer");
		deferredBuffer.Bind();
		ClearFramebuffer<ColorBuffer | DepthBuffer>();
		VAO& outerzone = VAOBank::Get("new_mesh");
		interzone.SetActiveShader();
		levelGeometry.Bind(outerzone);
		outerzone.Bind();
		outerzone.BindArrayBuffer(levelGeometry.vertex, 0);
		outerzone.BindArrayBuffer(Bank<ArrayBuffer>::Get("dummyInstance"), 1);
		interzone.SetVec3("shapeColor", glm::vec3(1.0, 1.0, 1.0));
		interzone.SetInt("checkUVs", debugFlags[CHECK_UVS]);
		interzone.SetTextureUnit("textureColor", Bank<Texture2D>::Get("blankTexture"), 0);
		//interzone.SetTextureUnit("textureColor", buffet.GetColor(), 0);
		//interzone.DrawElements(levelGeometry.index);
		interzone.MultiDrawElements(levelGeometry.indirect);

		//tickTockMan.Draw(guyMeshData, VAOBank::Get("new_mesh_single"), ShaderBank::Get("new_mesh_single"));
		management.Draw(guyMeshData, outerzone, interzone);

		interzone.SetActiveShader();
		outerzone.Bind();
		auto& buf = BufferBank::Get("player");
		auto meshs2 = playerModel;
		meshs2.scale *= 0.5f;
		auto meshs = meshs2.GetMatrixPair();
		buf.BufferData(std::to_array({ meshs.model, meshs.normal }));
		outerzone.BindArrayBuffer(buf, 1);
		playfield.Draw(interzone, outerzone, playerMesh2, playerModel);
		bobert.Draw();

		heWhoSleeps.Draw();

		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glCullFace(GL_FRONT);

		pointLightBuffer.Bind();
		ClearFramebuffer<ColorBuffer>();
		drawingVolumes.ExclusiveOperation([](auto& data) {Bank<ArrayBuffer>::Get("light_volume_mesh").BufferData(data, DynamicDraw); });
		// I don't know man this is too much
		if (featureToggle && false)
		{
			Shader& throne = ShaderBank::Get("light_volume_mesh");
			VAO& shadow = VAOBank::Get("light_volume_mesh");
			ArrayBuffer& cotillion = Bank<ArrayBuffer>::Get("light_volume_mesh");
			throne.SetActiveShader();
			shadow.Bind();
			shadow.BindArrayBuffer(sphereBuffer, 0);
			shadow.BindArrayBuffer(cotillion, 1);
			sphereIndicies.BindBuffer();
			throne.SetTextureUnit("gPosition", deferredBuffer.GetColorBuffer<0>(), 0);
			throne.SetTextureUnit("gNormal", deferredBuffer.GetColorBuffer<1>(), 1);
			if (featureToggle || true)
			{
				throne.DrawElementsInstanced<DrawType::Triangle>(sphereIndicies, cotillion);
			}
			else
			{
				// Possibility of working if locked to camera perspective but unsure
				shadow.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"), 0);
				solidCubeIndex.BindBuffer();
				throne.DrawElementsInstanced<DrawType::Triangle>(solidCubeIndex, cotillion);
			}
		}
		else
		{
			glCullFace(GL_BACK);
			Shader& throne = ShaderBank::Get("light_volume");
			VAO& shadow = VAOBank::Get("light_volume");
			ArrayBuffer& cotillion = Bank<ArrayBuffer>::Get("light_volume_mesh");
			throne.SetActiveShader();
			shadow.Bind();
			shadow.BindArrayBuffer(cotillion, 0);
			throne.SetVec3("cameraForward", cameraForward);
			throne.SetVec3("cameraPosition", localCamera);
			throne.SetTextureUnit("gPosition", deferredBuffer.GetColorBuffer<0>(), 0);
			throne.SetTextureUnit("gNormal", deferredBuffer.GetColorBuffer<1>(), 1);
			//throne.DrawArrayInstanced<DrawType::TriangleStrip>(Bank<ArrayBuffer>::Get("dummy"), cotillion);
			throne.DrawArrayInstanced<DrawType::TriangleFan>(Bank<ArrayBuffer>::Get("dummy2"), cotillion);
		}

		//throne.DrawElements<DrawType::Triangle>(sphereIndicies);
		//throne.DrawArrayInstanced<DrawType::TriangleStrip>(Bank<ArrayBuffer>::Get("dummy"), Bank<ArrayBuffer>::Get("lightVolume"));
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glCullFace(GL_BACK);
		// Do the actual deferred rendering
		BindDefaultFrameBuffer();

		Shader& interzone2 = ShaderBank::Get("combinePass");
		interzone2.SetActiveShader();
		//interzone2.SetInt("featureToggle", featureToggle);
		interzone2.SetVec3("lightPos", playerModel.translation);
		interzone2.SetVec3("lightDir", playerModel.rotation * glm::vec3(1.f, 0.f, 0.f));
		interzone2.SetTextureUnit("gPosition", deferredBuffer.GetColorBuffer<0>(), 0);
		interzone2.SetTextureUnit("gNormal", deferredBuffer.GetColorBuffer<1>(), 1);
		interzone2.SetTextureUnit("gColor", deferredBuffer.GetColorBuffer<2>(), 2);
		interzone2.SetTextureUnit("gLighting", pointLightBuffer.GetColor(), 3);
		interzone2.SetTextureUnit("gDepth", deferredBuffer.GetDepth(), 4);
		interzone2.DrawArray<DrawType::TriangleStrip>(4);

		//EnableGLFeatures<DepthTesting>();
		//EnableDepthBufferWrite();
	}
	CheckError();
	/* STICK FIGURE GUY */
	uniform.SetActiveShader();
	plainVAO.Bind();
	plainVAO.BindArrayBuffer(stickBuffer);

	glm::vec3 colors = glm::vec3(1, 0, 0);
	Model m22(glm::vec3(10, 0, 0));
	uniform.SetMat4("Model", m22.GetModelMatrix());
	uniform.SetVec3("color", colors);
	uniform.DrawElements<DrawType::LineStrip>(stickIndicies);

	DisableGLFeatures<FaceCulling>();

	//EnableGLFeatures<FaceCulling>();
	billboardShader.SetActiveShader();
	billboardVAO.Bind();
	billboardVAO.BindArrayBuffer(billboardBuffer, 0);
	billboardVAO.BindArrayBuffer(billboardMatrix, 1);
	billboardShader.SetTextureUnit("sampler", wallTexture, 0);
	// What?
	glm::vec3 radians = -glm::radians(cameraRotation);
	glm::mat4 cameraOrientation = glm::eulerAngleXYZ(radians.z, radians.y, radians.x);
	billboardShader.DrawArrayInstanced<DrawType::TriangleStrip>(billboardBuffer, billboardMatrix);
	EnableGLFeatures<FaceCulling>();

	if (debugFlags[DYNAMIC_TREE])
	{
		uniform.SetActiveShader();
		glm::vec3 blue(0, 0, 1);
		plainVAO.Bind();
		plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"));
		uniform.SetVec3("color", glm::vec3(1, 0.65, 0));
		for (auto& box : dynamicTreeBoxes)
		{
			auto d = box.GetModel();
			d.scale *= 0.99f;
			uniform.SetMat4("Model", d.GetModelMatrix());
			uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
		}
	}
	uniform.SetActiveShader();
	plainVAO.Bind();
	plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"));
	uniform.SetVec3("color", glm::vec3(1, 0.65, 0));
	const OBB& target = Bank<OBB>::Get("NoGoZone");
	uniform.SetMat4("Model", target.GetModelMatrix());
	uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);

	const glm::mat3 axes22 = glm::mat3_cast(playerModel.rotation);
	uniform.SetActiveShader();
	plainVAO.Bind();
	plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"));
	uniform.SetVec3("color", glm::vec3(0.f, 0.f, 1.f));
	glLineWidth(1.f);
	glm::vec3 bulletPath = glm::normalize(axes22[0] * 100.f + playfield.GetVelocity());
	glm::vec3 position = playerModel.translation + bulletPath * 10.f;
	Model model{ position, playerModel.rotation };
	model.scale = glm::vec3(0.25f);
	uniform.SetMat4("Model", model.GetModelMatrix());
	uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
	for (int i = 0; i < 5; i++)
	{
		model.translation += 10.f * bulletPath;
		uniform.SetMat4("Model", model.GetModelMatrix());
		uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
	}
	uniform.SetMat4("Model", management.GetOBB().GetModelMatrix());
	uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);

	// Debugging staticBoxes
	if (debugFlags[TIGHT_BOXES])
	{
		OBB localCopy = Player::Box;
		localCopy.Rotate(playerModel.GetModelMatrix());
		uniform.SetMat4("Model", localCopy.GetModelMatrix());
		uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
	}

	// Cubert
	uniform.SetActiveShader();
	plainVAO.Bind();
	plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"));
	//uniform.SetMat4("Model", dumbBox.GetModelMatrix());
	//uniform.DrawElements<DrawType::Triangle>(solidCubeIndex);

	ship.SetActiveShader();
	meshVAO.Bind();
	//meshVAO.BindArrayBuffer(guyBuffer);
	//meshVAO.BindArrayBuffer(guyMeshData.vertex);
	//guyMeshData.index.BindBuffer();

	CheckError();

	{
		// TODO: move this elsewhere
		FeatureFlagPush<Blending> _blend;
		FeatureFlagPush<FaceCulling, false> _blend2;
		DisableDepthBufferWrite();
		Shader& foolish = ShaderBank::Get("Shielding");
		VAO& vao = VAOBank::Get("simple_mesh_instance");
		ArrayBuffer& buffer = Bank<ArrayBuffer>::Get("shieldPos");
		foolish.SetActiveShader();
		vao.Bind();
		vao.BindArrayBuffer(sphereBuffer, 0);
		vao.BindArrayBuffer(buffer, 1);
		//sphereBuffer.BindBuffer();
		sphereIndicies.BindBuffer();
		foolish.SetTextureUnit("textureIn", buffet.GetColor(), 0);
		//foolish.SetTextureUnit("textureIn", Bank<Texture2D>::Get("flma"), 0);
		Model maudlin;
		//maudlin.translation = glm::vec3(0, 60.f, 0.f);
		maudlin.scale = glm::vec3(4.f * glm::compMax(ClockBrain::Collision.GetScale()));
		foolish.SetMat4("modelMat", maudlin.GetModelMatrix());
		foolish.SetMat4("normalMat", glm::mat4(1.f));
		foolish.SetInt("FeatureToggle", featureToggle);
		//foolish.DrawElements(sphereIndicies);
		foolish.DrawElementsInstanced<DrawType::Triangle>(sphereIndicies, buffer);
		EnableDepthBufferWrite();
	}

	Model defaults(playerModel);

	defaults.translation = glm::vec3(10, 10, 0);
	defaults.rotation = glm::quat(0.f, 0.f, 0.f, 1.f);
	defaults.scale = glm::vec3(0.5f);
	ship.SetMat4("modelMat", defaults.GetModelMatrix());
	ship.SetMat4("normalMat", defaults.GetNormalMatrix());
	groovy.Draw(ship);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	EnableGLFeatures<Blending>();
	DisableDepthBufferWrite();
	// TODO: Maybe look into this https://www.opengl.org/archives/resources/code/samples/sig99/advanced99/notes/node20.html
	decalShader.SetActiveShader();
	texturedVAO.Bind();
	texturedVAO.BindArrayBuffer(decals);
	decalShader.SetTextureUnit("textureIn", texture, 0);
	decalShader.DrawArray<DrawType::Triangle>(decals);
	EnableDepthBufferWrite();
	DisableGLFeatures<Blending>();
	/*
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		Shader& shaderRef = uniform;
		VAO& vaoRef = plainVAO;//VAOBank::Get("uniformInstance");
		shaderRef.SetActiveShader();
		vaoRef.Bind();
		vaoRef.BindArrayBuffer(plainCube, 0);
		//vaoRef.BindArrayBuffer(levelOutline, 1);
		//shaderRef.SetMat4("Model2", glm::translate(glm::mat4(1.f), glm::vec3(0.f, 10.f, 0.f)));
		shaderRef.SetMat4("Model", bulletBox.GetModelMatrix());
		shaderRef.DrawElements<DrawType::Triangle>(solidCubeIndex);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}*/

	// Level outline
	if (featureToggle)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		DisablePushFlags(DepthTesting | FaceCulling);
		Shader& shaderRef = uniform;
		VAO& vaoRef = plainVAO;//VAOBank::Get("uniformInstance");
		shaderRef.SetActiveShader();
		vaoRef.Bind();
		vaoRef.BindArrayBuffer(levelOutline, 0);
		//vaoRef.BindArrayBuffer(levelOutline, 1);
		//shaderRef.SetMat4("Model2", glm::translate(glm::mat4(1.f), glm::vec3(0.f, 10.f, 0.f)));
		shaderRef.SetMat4("Model", glm::mat4(1.f));
		shaderRef.DrawArray<DrawType::Triangle>(levelOutline);
		//shaderRef.DrawElementsInstanced<DrawType::Lines>(cubeOutlineIndex, levelOutline);
		//shaderRef.DrawElementsInstanced<DrawType::Triangle>(solidCubeIndex, levelOutline);
	}
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//meshVAO.BindArrayBuffer(guyBuffer2);
	CheckError();

	//normalDebris.SetActiveShader();
	//normalDebris.SetTextureUnit("normalMapIn", normalMap);
	trashMan.Draw(debris);
	glLineWidth(1.f);

	basic.SetActiveShader();
	meshVAO.Bind();
	meshVAO.BindArrayBuffer(sphereBuffer);
	sphereIndicies.BindBuffer();
	basic.SetVec4("Color", glm::vec4(2.f, 204.f, 254.f, 250.f) / 255.f);
	basic.SetMat4("Model", magnetic.GetMatrix(playerModel.translation));
	basic.DrawElements<DrawType::Lines>(sphereIndicies);
	CheckError();
	glDepthMask(GL_TRUE);
	// Albert
	//glPatchParameteri(GL_PATCH_VERTICES, 3);
	texturedVAO.Bind();
	texturedVAO.BindArrayBuffer(albertBuffer);
	{
		Shader& dither = ShaderBank::Get("dither");
		dither.SetActiveShader();
		dither.SetTextureUnit("ditherMap", wallTexture, 1);
		dither.SetTextureUnit("textureIn", texture, 0);
		dither.SetMat4("Model", dumbBox.GetModelMatrix());
		dither.SetVec3("color", glm::vec3(0, 1, 0));
		dither.SetVec3("lightColor", glm::vec3(1.f, 1.f, 1.f));
		dither.SetVec3("lightPos", glm::vec3(5.f, 1.5f, 0.f));
		dither.SetVec3("viewPos", cameraPosition);
		//dither.DrawArray<DrawType::Triangle>(36);
		//dither.DrawArray<DrawType::Patches>(albertBuffer);
	}


	plainVAO.Bind();
	plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"));
	uniform.SetActiveShader();
	uniform.SetMat4("Model", dumbBox.GetModelMatrix());
	//uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);

	// Drawing of the rays
	plainVAO.BindArrayBuffer(rayBuffer);
	Model bland;
	uniform.SetMat4("Model", bland.GetModelMatrix());
	glLineWidth(15.f);
	//uniform.DrawArray<DrawType::Lines>(rayBuffer);

	trails.SetActiveShader();
	colorVAO.Bind();
	DisableGLFeatures<FaceCulling>();
	EnableGLFeatures<Blending>();

	trails.SetVec3("Color", glm::vec3(2.f, 204.f, 254.f) / 255.f);
	colorVAO.BindArrayBuffer(leftBuffer);
	trails.DrawArray<DrawType::TriangleStrip>(leftBuffer);
	colorVAO.BindArrayBuffer(rightBuffer);
	trails.DrawArray<DrawType::TriangleStrip>(rightBuffer);
	EnableGLFeatures<FaceCulling>();
	DisableGLFeatures<Blending>();

	glLineWidth(1.f);

	//tickTockMan.Draw(guyMeshData, VAOBank::Get("new_mesh"), ShaderBank::Get("new_mesh"));
	//debris.SetActiveShader();
	//management.Draw(guyMeshData, meshVAO, debris);

	CheckError();

	engine.SetActiveShader();
	VAOBank::Get("engineInstance").Bind();
	VAOBank::Get("engineInstance").BindArrayBuffer(exhaustBuffer);
	engine.SetUnsignedInt("Time", static_cast<unsigned int>(gameTicks & std::numeric_limits<unsigned int>::max()));
	engine.SetUnsignedInt("Period", 150);
	engine.DrawArrayInstanced<DrawType::Triangle>(Bank<ArrayBuffer>::Get("dummyEngine"), exhaustBuffer);
	//EnableGLFeatures<DepthTesting>();
	CheckError();
	// Sphere drawing
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (bulletMesh.rawIndirect[1].instanceCount > 0)
	{
		Shader& bulletShader = ShaderBank::Get("bulletShader");
		bulletShader.SetActiveShader();
		bulletMesh.Bind(bulletVAO);
		bulletVAO.BindArrayBuffer(bulletMats, 1);
		bulletShader.MultiDrawElements(bulletMesh.indirect);
		{
			
			Shader& shaderRef = ShaderBank::Get("uniformInstance");
			VAO& vaoRef = VAOBank::Get("uniformInstance");
			shaderRef.SetActiveShader();
			vaoRef.Bind();
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			vaoRef.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"), 0);
			vaoRef.BindArrayBuffer(bulletMats, 1);
			shaderRef.SetMat4("Model2", glm::mat4(1.f));
			//shaderRef.DrawElementsInstanced<DrawType::Lines>(cubeOutlineIndex, bulletMats2);

			vaoRef.BindArrayBuffer(Bank<ArrayBuffer>::Get("bulletImpacts"), 1);
			shaderRef.DrawElementsInstanced<DrawType::Lines>(cubeOutlineIndex, Bank<ArrayBuffer>::Get("bulletImpacts"));
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			
		}
	}
	Model sphereModel{};
	sphereModel.scale = glm::vec3(4.f, 4.f, 4.f);
	sphereModel.translation = glm::vec3(0, 0, 0);
	Model lightModel;
	lightModel.translation = glm::vec3(4, 0, 0);
	lightModel.scale = glm::vec3(2.2f);//glm::vec3(2.2, 2.2, 1.1);

	sphereModel.translation += glm::vec3(0, 1, 0) * glm::sin(glm::radians(idleFrameCounter * 0.25f)) * 3.f;
	stencilTest.SetActiveShader();

	// TODO: Make something to clarify the weirdness of the stencil function
	// Stuff like the stencilOp being in order: Stencil Fail(depth ignored), Stencil Pass(Depth Fail), Stencil Pass(Depth Pass)
	// And stencilFunc(op, ref, mask) does the operation on a stencil value K of: (ref & mask) op (K & mask)
	
	/*
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); // Read from Default
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, experiment.GetFrameBuffer());
	glBlitFramebuffer(0, 0, Window::Width, Window::Height, 0, 0, Window::Width / 2, Window::Height / 2, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	experiment.Bind();
	glClearStencil(0x00);
	glClear(GL_STENCIL_BUFFER_BIT);

	// All shadows/lighting will be in this post-processing step based on stencil value

	//////  Shadow volume
	glDepthMask(GL_FALSE); // Disable writing to the depth buffer

	// Stencil tests must be active for stencil shading to be used
	// Depth testing is required to determine what faces the volumes intersect
	EnableGLFeatures<StencilTesting | DepthTesting>();
	// We are using both the front and back faces of the model, cannot be culling either
	DisableGLFeatures<FaceCulling>();

	// To make the inverse kind of volume (shadow/light), simply change the handedness of the system AND BE SURE TO CHANGE IT BACK
	//glFrontFace((featureToggle) ? GL_CCW : GL_CW);
	//glFrontFace(GL_CCW);
	// Stencil Test Always Passes
	glStencilFunc(GL_ALWAYS, 0, 0xFF);
	
	// Back Faces increment the stencil value if they are behind the geometry, ie the geometry
	// is inside the volume
	glStencilOpSeparate(GL_BACK, GL_KEEP, GL_INCR_WRAP, GL_KEEP);
	// Front faces decrement if they are behind geometry, so that surfaces closer to the camera
	// than the volume are not incorrectly shaded by volumes that don't touch it
	glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_DECR_WRAP, GL_KEEP);

	// Drawing of the appropriate volumes
	stencilTest.SetMat4("Model", sphereModel.GetModelMatrix());
	meshVAO.BindArrayBuffer(sphereBuffer);
	//stencilTest.DrawElements<DrawType::Triangle>(sphereIndicies);

	plainVAO.BindArrayBuffer(plainCube);
	//for (int i = 0; i < 4; i++)
	int count = 0;
	for (auto& following : followers)
	{
		lightModel.translation = following.first.GetPosition();
		lightModel.scale = glm::vec3(2.3f + float(count++));
		//lightModel.Translate(glm::vec3(2 + i * glm::cos(frameCounter / 100.f), 0, 0));
		stencilTest.SetMat4("Model", lightModel.GetModelMatrix());
		stencilTest.DrawElementsMemory<DrawType::Triangle>(cubeIndicies);
	}
	
	// Clean up
	EnableGLFeatures<FaceCulling>();
	//DisableGLFeatures<StencilTesting>();
	glFrontFace(GL_CCW);
	//glDepthMask(GL_TRUE); // Allow for the depth buffer to be written to
	//////  Shadow volume End
	BindDefaultFrameBuffer();
	*/
	CheckError();
	//GL_ARB_shader_stencil_export
	// TODO: Figure this out
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

	DisableGLFeatures<FaceCulling>();
	glDepthFunc(GL_LEQUAL);
	skyBox.SetActiveShader();
	plainVAO.Bind();
	plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"), 0);
	skyBox.SetTextureUnit("skyBox", sky);
	//skyBox.DrawElements<DrawType::Triangle>(solidCubeIndex);
	EnableGLFeatures<FaceCulling>();

	basic.SetActiveShader();
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	{
		//heWhoSleeps.model.rotation = ForwardDir(glm::vec3(0.f, 1.f, 0.f), glm::vec3(-1.f, 0.f, 0.f));
		//heWhoSleeps.openStyle = Door::Type::Square;
		/*
		heWhoSleeps.model.rotation = glm::quat(glm::vec3(glm::sin(glm::radians((float)gameTicks)), 0.f, 
			glm::cos(glm::radians((float)gameTicks * 3.2f))));
		*/
		std::vector<glm::vec3> fdso;
		auto sd = heWhoSleeps.GetTris();
		for (const Triangle& p : sd)
		{
			for (const auto& sdf : p.GetPointArray())
				fdso.push_back(sdf);
		}
		
		rayBuffer.BufferSubData(fdso);
	}
	plainVAO.Bind();
	plainVAO.BindArrayBuffer(rayBuffer);
	//VAOBank::Get("muscle").Bind();
	//VAOBank::Get("muscle").BindArrayBuffer(guyMeshData.vertex);
	basic.SetMat4("Model", glm::mat4(1.f));
	basic.SetVec4("Color", glm::vec4(1.f));
	basic.DrawArray(rayBuffer);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//basic.DrawArray<DrawType::Points>(guyMeshData.vertex);


	CheckError();
	/*
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
	nineSliced.Bind();
	nineSliced.BindArrayBuffer(ui_tester_buffer);
	//nineSlicer.DrawArrayInstanced<DrawType::TriangleStrip>(4, 9);
	*/

	
	uiRectTexture.SetActiveShader();
	
	auto& colored = buffet.GetColor();
	uiRectTexture.SetTextureUnit("image", colored, 0);
	glm::vec4 loc = glm::vec4((Window::Width - colored.GetWidth()) / 2, (Window::Height - colored.GetHeight()) / 2,
		colored.GetWidth(), colored.GetHeight());
	uiRectTexture.SetVec4("rectangle", loc);
	//uiRect.DrawArray<DrawType::TriangleStrip>(4);
	/*
	uiRectTexture.SetTextureUnit("image", (buttonToggle) ? buttonA : buttonB, 0);
	uiRectTexture.SetVec4("rectangle", buttonRect);
	uiRect.DrawArray<DrawType::TriangleStrip>(4);

	uiRectTexture.SetTextureUnit("image", help.GetTexture(), 0);
	uiRectTexture.SetVec4("rectangle", help.GetRect());
	uiRect.DrawArray<DrawType::TriangleStrip>(4);

	uiRectTexture.SetTextureUnit("image", normalMap);
	uiRectTexture.SetVec4("rectangle", { 0, 0, normalMap.GetSize()});
	//uiRect.DrawArray<DrawType::TriangleStrip>(4);
	CheckError();
	DisableGLFeatures<FaceCulling>();
	DisableGLFeatures<Blending>();
	*/
	EnableGLFeatures<Blending>();
	// Debug Info Display
	fontShader.SetActiveShader();
	fontVAO.Bind();
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
	
	/*
	frameShader.SetActiveShader();
	frameShader.SetTextureUnit("normal", depthed.GetColorBuffer<1>(), 0);
	frameShader.SetTextureUnit("depth", depthed.GetDepth(), 1);
	frameShader.SetFloat("zNear", zNear);
	frameShader.SetFloat("zFar", zFar);
	frameShader.SetInt("zoop", 0);
	frameShader.DrawArray<TriangleStrip>(4);
	*/
	glStencilMask(0xFF);

	glLineWidth(1.f);
	widget.SetActiveShader();
	widget.DrawArray<DrawType::Lines>(6);
	
	EnableGLFeatures<DepthTesting | StencilTesting | FaceCulling>();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	auto end = std::chrono::high_resolution_clock::now();
	displayTime = end - displayStartTime;
	displayStartTime = end;
	glFinish();
	end = std::chrono::high_resolution_clock::now();
	renderDelay = end - displayStartTime;
}

// TODO: Mech suit has an interior for the pilot that articulates seperately from the main body, within the outer limits of the frame
// Like it's a bit pliable

static long long maxTickTime;
static long long averageTickTime;
static glm::vec3 targetAngles{0.f};

CircularBuffer<ColoredVertex, 256> leftCircle, rightCircle;

std::binary_semaphore setPosSemaphore{ 0 };

// This function *is* allowed to touch OpenGL memory, as it is on the same thread. If another one does it then OpenGL breaks
void idle()
{
	static auto lastIdleStart = std::chrono::high_resolution_clock::now();

	static TimerAverage<300> displayTimes, idleTimes, renderTimes;
	static TimerAverage<300, float> frames;
	static CircularBuffer<float, 200> fpsPlot;
	static unsigned long long displaySimple = 0, idleSimple = 0;

	idleFrameCounter++;
	const TimePoint idleStart = std::chrono::high_resolution_clock::now();
	const TimeDelta delta = idleStart - lastIdleStart;

	const float timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(delta).count();

	float averageFps = frames.Update(1.f / timeDelta);
	
	long long averageIdle = idleTimes.Update(idleTime.count() / 1000);
	long long averageDisplay = displayTimes.Update(displayTime.count() / 1000);
	long long averageRender = renderTimes.Update(renderDelay.count() / 1000);

	fpsPlot.Push(timeDelta * 1000.f);
	static bool disableFpsDisplay = true;

	if (disableFpsDisplay)
	{
		ImGui::Begin("Metrics", &disableFpsDisplay);
		auto frames = fpsPlot.GetLinear();
		ImGui::PlotLines("##2", frames.data(), static_cast<int>(frames.size()), 0, "Frame Time", 0.f, 5.f, ImVec2(100, 100));
		ImGui::SameLine(); ImGui::Text(std::format("(ms): {:2.3}", 1000.f / averageFps).c_str());
		ImGui::End();
	}
	if (timeDelta > 0.01)
	{
		Log("SPIKE: " << timeDelta);
	}

	// "Proper" input handling
	// These functions should be moved to the gametick loop, don't want to over-poll the input device and get weird
	Gamepad::Update();
	Mouse::UpdateEdges();
	boardState = Input::UpdateStatus();

	help.MouseUpdate();
	Input::UIStuff();
	Input::DisplayInput();
	if (!Input::ControllerActive())
	{
		float tilt = 0.f;
		if (keyState['Q'] && !keyState['E']) tilt = 1.f;
		if (keyState['E'] && !keyState['Q']) tilt = -1.f;
		targetAngles.z = tilt * 0.5f;
		glm::vec3 adjustment = (shiftHeld) ? glm::vec3(0.5f) : glm::vec3(1.f);
		boardState.heading = glm::vec4(playerSpeedControl, targetAngles * adjustment);
		boardState.fireButton = keyState['T'];
		boardState.cruiseControl = keyState['Y'];
		boardState.popcornFire = Mouse::CheckButton(Mouse::ButtonLeft);

		boardState.movement = glm::vec3(0.f);
		boardState.movement.x +=  1.f * keyState['W'];
		boardState.movement.x += -1.f * keyState['S'];
		boardState.movement.y +=  1.f * keyState['Z'];
		boardState.movement.y += -1.f * keyState['X'];
		boardState.movement.z += -1.f * keyState['A'];
		boardState.movement.z +=  1.f * keyState['D'];
		boardState.rotation = glm::yzw(boardState.heading);

		boardState.zoomZoom = keyState['R']; // Make this shift
	}
	else
	{
		if (Input::Gamepad::CheckRisng(Input::Gamepad::DPadUp))
		{
			playerSpeedControl += 0.1f;
		}
		if (Input::Gamepad::CheckRisng(Input::Gamepad::DPadDown))
		{
			playerSpeedControl -= 0.1f;
		}
		float thrust = Input::Gamepad::CheckAxes(0).y;
		// TOOD: Account for this in a better manner
		if (glm::abs(thrust) > 0.125) // TODO: Deadzone constants
		{
			//playerSpeedControl += glm::sign(-thrust) * timeDelta * 0.25f;
		}
		float turn = 0.f;
		if (Input::Gamepad::CheckButton(Input::Gamepad::LeftBumper))  turn -= 1.f;
		if (Input::Gamepad::CheckButton(Input::Gamepad::RightBumper)) turn += 1.f;
		playerSpeedControl = glm::clamp(playerSpeedControl, 0.f, 1.f);
		boardState.heading.y = -Input::Gamepad::CheckAxes(1).x;//-input.axes[2];
		boardState.heading.z = Input::Gamepad::CheckAxes(1).y;
		boardState.heading.w = turn;
		boardState.heading.x = playerSpeedControl;
		boardState.popcornFire = Input::Gamepad::CheckButton(Input::Gamepad::A);
		boardState.popcornFire |= Input::Gamepad::CheckAxes(2).x > 0.f;

		glm::vec2 bad = Input::Gamepad::CheckAxes(0);
		glm::vec2 sig = glm::sign(bad);
		if (glm::abs(bad).x < 0.1)
		{
			bad.x = 0;
		}
		if (glm::abs(bad).y < 0.1)
		{
			bad.y = 0;
		}

		boardState.movement.x = -bad.y;
		boardState.movement.z = bad.x;
		boardState.movement.y = 0.f;
		boardState.movement.y += 1.f * Input::Gamepad::CheckButton(Input::Gamepad::A);
		boardState.movement.y -= 1.f * Input::Gamepad::CheckButton(Input::Gamepad::B);

		boardState.rotation = glm::yzw(boardState.heading);

		// Something weird with this and the cruise control button for some reason
		boardState.fireButton = Input::Gamepad::CheckAxes(2).y > 0.f;
		boardState.cruiseControl = Input::Gamepad::CheckButton(Input::Gamepad::X);
		if (Input::Gamepad::CheckButton(Input::Gamepad::BackButton))
		{
			shouldClose = true;
			glfwSetWindowShouldClose(windowPointer, true);
		}

	}
		
	// End of input handling
	if (reRenderText && letters.str().size() > 0)
	{
		reRenderText = false;
		playerTextEntry = fonter.Render(letters.str(), glm::vec4(1.f, 0.f, 0.f, 1.f));
		std::stringstream().swap(letters);
	}

	//followed.Update();

	if (debugFlags[DYNAMIC_TREE])
	{
		dynamicTreeBoxes = Level::GetTriangleTree().GetBoxes();
	}
	std::vector<glm::mat4> matx;

	for (auto& follow : followers)
	{
		follow.Update();
		matx.push_back(follow.GetModelMatrix());
	}
	billboardMatrix.BufferData(matx);

	const Model playerModel = playfield.GetModel();
	static bool flippyFlop = false;
	if (idleFrameCounter % 10 == 0)
	{
		glm::mat3 playerLocal = static_cast<glm::mat3>(playerModel.rotation);
		glm::vec3 forward = playerLocal[0];
		glm::vec3 left = playerLocal[2];
		left *= 0.5f;
		glm::vec3 local = playerModel.translation;
		local -= forward * 0.55f;
		glm::vec3 upSet = playerLocal[1] * 0.05f;
		if (flippyFlop)
		{

		}
		flippyFlop = !flippyFlop;
		leftCircle.Push(ColoredVertex{  local - left,  upSet + left * 0.2f});
		leftCircle.Push(ColoredVertex{  local - left, -upSet - left * 0.2f});
		rightCircle.Push(ColoredVertex{ local + left,  upSet - left * 0.2f });
		rightCircle.Push(ColoredVertex{ local + left, -upSet + left * 0.2f });

		leftBuffer.BufferData(leftCircle.GetLinear());
		rightBuffer.BufferData(rightCircle.GetLinear());
	}

	static glm::vec3 lastCheckedPos = glm::vec3(0.f, 3.f, 0.f);
	static float lastCheckedDistance = 99;
	static std::size_t lastCheckedTick = 0;
	// TODO: have an explicit "only activate once per game tick" zone
	if (gameTicks % 128 == 0 && gameTicks != lastCheckedTick)
	{
		glm::vec3 localPos = playerModel.translation;
		lastCheckedDistance = glm::distance(lastCheckedPos, localPos);
		lastCheckedPos = localPos;
		lastCheckedTick = gameTicks;
		if (gameTicks % 256 == 0)
		{
			glm::vec3 color = glm::abs(glm::ballRand(1.f));
			color.z = glm::clamp(color.z, 0.42f, 0.8f);
			color = glm::vec3(0.9f, 0.9f, 0.9f);
			//lightingBuffer.BufferSubData(glm::vec4(glm::rgbColor(color), 0.f), 0);
			lightingBuffer.BufferSubData(glm::vec4(color, 0.f), 0);
		}
	}
	std::vector<glm::vec3> rays;
	Model rayLocal = playerModel;
	glm::mat3 localSpace(rayLocal.rotation);
	rays.push_back({});
	rays.push_back(localSpace[0]);
	rays.push_back({});
	rays.push_back(playfield.GetVelocity());

	for (auto& lco : rays)
	{
		lco += rayLocal.translation + localSpace[1];
	}

	// Better bullet drawing
	{
		bulletMatricies.ExclusiveOperation([&](std::vector<glm::mat4>& mats) 
			{
				bulletMesh.rawIndirect[0].instanceCount = 0;
				bulletMesh.rawIndirect[1].instanceCount = static_cast<GLuint>(mats.size());
				bulletMesh.indirect.BufferSubData(bulletMesh.rawIndirect);
				bulletMats.BufferData(mats);
			}
		);

		bulletImpacts.ExclusiveOperation([&](std::vector<glm::mat4>& mats)
			{
				Bank<ArrayBuffer>::Get("bulletImpacts").BufferData(mats);
			}
		);
		management.UpdateMeshes();
		shieldPos.ExclusiveOperation([&](std::vector<glm::vec3>& bufs)
			{
				Bank<ArrayBuffer>::Get("shieldPos").BufferData(bufs);
			}
		);
	}

	Parallel::SetStatus(!keyState['P']);

	std::stringstream buffered;
	buffered << playfield.GetVelocity() << ":" << glm::length(playfield.GetVelocity());
	buffered << "\n" << playfield.GetModel().translation;
	buffered << "\nFeatureToggle: " << std::boolalpha << featureToggle << "\nFull Calculations: " << debugFlags[FULL_CALCULATIONS];
	{
		buffered << '\n' << Level::GetBulletTree().size();
		// Only for debugging lights
		/*
		glm::vec3 localCamera = cameraPosition;
		const glm::mat3 axes(playerModel.rotation);
		const glm::vec3 velocity = playfield.GetVelocity();

		localCamera = glm::vec3(4.f, -2.5f, 0.f);
		localCamera = (playerModel.rotation * aboutTheShip) * localCamera;
		localCamera += playerModel.translation;
		localCamera -= velocity / 20.f;
		const glm::vec3 cameraFocus = playerModel.translation + axes[0] * 10.f;
		const glm::vec3 cameraForward = glm::normalize(cameraFocus - localCamera);

		glm::vec3 loc(testCameraPos);
		glm::vec3 direction = localCamera - loc;
		float length = glm::length(direction);
		glm::vec3 unit = glm::normalize(direction);
		glm::vec3 newPos = loc;
		buffered << std::format("\nDot: {}\nDistance: {}\n", glm::dot(unit, cameraForward), length);
		if (glm::dot(unit, cameraForward) > 0)
		{
			newPos += (glm::dot(unit, cameraForward) * length * 1.1f) * cameraForward;
		}
		glm::vec3 newLoc = localCamera - newPos;
		buffered << std::format("Dot: {}\nDistance: {}", (glm::dot(glm::normalize(newLoc), cameraForward)), glm::length(newLoc));
		*/
	}
	Level::SetInterest(management.GetPos());
	
	constexpr auto formatString = "FPS:{:7.2f}\nTime:{:4.2f}ms\nIdle:{}ns\nDisplay:\n-Concurrent: {}ns\
		\n-GPU Block Time: {}ns\nAverage Tick Length:{}ns\nMax Tick Length:{:4.2f}ms\nTicks/Second: {:7.2f}\n{}";

	std::string formatted = std::format(formatString, averageFps, 1000.f / averageFps, averageIdle, 
		averageDisplay, averageRender, averageTickTime, maxTickTime / 1000.f, gameTicks / glfwGetTime(), buffered.str());

	fonter.GetTextTris(textBuffer, 0, 0, formatted);

	std::ranges::copy(keyState, std::begin(keyStateBackup));

	decalVertex.ExclusiveOperation([&](auto& ref)
		{
			decals.BufferData(ref, StaticDraw);
		}
	);
	managedProcess.FillBuffer(exhaustBuffer);
	trashMan.FillBuffer();

	const auto endTime = std::chrono::high_resolution_clock::now();
	idleTime = endTime - idleStart;
	lastIdleStart = idleStart;
}

// *Must* be in a separate thread
void gameTick()
{
	using namespace std::chrono_literals;
	constexpr std::chrono::duration<long double> tickInterval = 0x1.p-7s;
	TimePoint lastStart = std::chrono::steady_clock::now();
	TimerAverage<300> gameTickTime;
	Level::ResetCurrentTick();
	do
	{
		const TimePoint tickStart = std::chrono::steady_clock::now();
		const TimeDelta interval = tickStart - lastStart;
		Capsule silly{ groovy.GetBounding() };

		const Frustum localFrust = GetFrustum(playfield.GetModel());

		// Bullet stuff;
		std::vector<glm::mat4> inactive, blarg;

		std::vector<LightVolume> volumes{ constantLights };
		// TODO: Combine these with a special function with an enum return value
		Level::GetBulletTree().for_each([&](Bullet& local)
			{
				if (!local.IsValid())
				{
					return false;
				}
				glm::vec3 previous = local.transform.position;
				if (!debugFlags[FREEZE_GAMEPLAY])
				{
					local.Update();
				}
				const AABB endState = local.GetAABB();

				if (!localFrust.Overlaps(Sphere(endState.GetCenter(), glm::compMax(endState.Deviation()))))
				{
					inactive.push_back(local.GetModel().GetModelMatrix());
					if (local.lifeTime > 10)
					{
						volumes.push_back({ glm::vec4(local.transform.position, 15.f), glm::vec3(1.f, 1.f, 0.f), glm::vec3(1.f, 0.f, 0.05f) });
					}
				}
				return previous != local.transform.position;
			});

		management.Update();

		auto tmep = bobert.GetPoints(management.GetRawPositions());
		std::vector<glm::vec3> shieldPoses;
		// This is bad and should be moved to the shield generator class
		for (glm::vec3 point : tmep)
		{
			Sphere spoke(point, 10.f);
			if (Level::GetBulletTree().QuickTest(spoke.GetAABB()))
			{
				shieldPoses.push_back(point);
			}
			volumes.push_back({ glm::vec4(point, 10.f), glm::vec3(120.f,204.f,226.f) / 255.f, glm::vec3(1.f, 0.5f, 0.05f) });
		}
		shieldPos.Swap(tmep);
		std::size_t removedBullets = Level::GetBulletTree().EraseIf([&](Bullet& local) 
			{
				if (!local.IsValid())
				{
					return true;
				}
				if (local.team == 0)
				{
					for (glm::vec3 point : shieldPoses)
					{
						glm::vec3 forward = (local.transform.rotation * glm::vec3(local.speed, 0.f, 0.f)) * Tick::TimeDelta;
						LineSegment segmentation{ local.transform.position - forward, local.transform.position + forward };
						Capsule flipper(segmentation, 0.1f);
						Collision hit{};
						Sphere bogus{ point, 4.f * glm::compMax(ClockBrain::Collision.GetScale()) };
						if (flipper.Intersect(bogus, hit))
						{
							if (!(bogus.SignedDistance(segmentation.A) < 0 && bogus.SignedDistance(segmentation.B) < 0))
							{
								//Log("Blown'd up");
								return true;
							}
						}
						/*
						float distance = glm::distance(point, local.transform.position);
						if (9.5f < distance && distance <= 10.f)
						{
							Level::SetExplosion(local.transform.position);
							return true;
						}
						*/
					}
				}
				OBB transformedBox = local.GetOBB();
				//blarg.push_back(transformedBox.GetModelMatrix());
				//blarg.push_back(transformedBox.GetAABB().GetModelMatrix());
				
				for (const auto& currentTri : Level::GetTriangleTree().Search(transformedBox.GetAABB()))
				{
					if (DetectCollision::Overlap(transformedBox, *currentTri))
					{
						// Don't let enemy decals clog things up 
						if (local.team != 0)
							return true;
						// TODO: change this so that the output vector isn't the big list so the actual generation of the decals
						// can be parallelized, with only the copying needing sequential access
						// If no decals were generated, then it didn't 'precisely' overlap any of the geometry, and as
						// generating decals also requires a OctTreeSearch, escape the outer one.
						if (decalVertex.ExclusiveOperation(
							[&](auto& ref)
							{
								//QuickTimer _time("Decal Generation");
								// TODO: Not completely pleased with this, which triangle is hit first has a big impact on the 
								// resulting decal, average the normal of all affected tris? I don't know yet
								float planeDistance = currentTri->GetPlane().Facing(transformedBox.GetCenter());
								glm::vec3 newCenter = transformedBox.GetCenter() - currentTri->GetNormal() * planeDistance;
								OBB sigma(Model(newCenter, ForwardDir(-currentTri->GetNormal(),
									local.transform.rotation * glm::vec3(0.f, 1.f, 0.f)),
									Bullet::Collision.GetScale() * glm::vec3(1.5f, BulletDecalScale, BulletDecalScale)));
								blarg.push_back(sigma.GetModelMatrix());
								if (Decal::GetDecal(sigma, Level::GetTriangleTree(), ref).size() == 0)
								{
									// Possibly helps things, but I'm not completely sure
									Log("Decal Failed");
									return false;
								}
								else
								{
									return true;
								}
							}
						))
						{
							// Decals generated -> must remove the bullet
							return true;
						}
						break;
					}
				}
				return false; 
			}
		);
		if (removedBullets > 0)
		{
			Level::GetBulletTree().UpdateStructure();
		}
		// Maybe this is a "better" method of syncing stuff than the weird hack of whatever I had before
		bulletMatricies.Swap(inactive);
		if (blarg.size() > 0)
		{
			bulletImpacts.Swap(blarg);
		}

		drawingVolumes.Swap(volumes);

		heWhoSleeps.Update();

		// Gun animation
		if (flubber.IsFinished())
		{
			flubber.Start(gameTicks);
		}
		
		playfield.Update(boardState);
		const Model playerModel = playfield.GetModel();
		if (Level::NumExplosion() > 0)
		{
			for (glm::vec3 copy : Level::GetExplosion())
			{
				for (int i = 0; i < 20; i++)
				{
					managedProcess.AddExhaust(copy + glm::ballRand(0.25f), glm::sphericalRand(5.f), 256);
				}
				for (int i = 0; i < 5; i++)
				{
					glm::vec3 velocity = glm::ballRand(5.f);
					if (glm::length(velocity) < 2.5f)
					{
						velocity *= 2.5f;
					}
					glm::vec3 center = glm::ballRand(0.25f);
					trashMan.AddDebris(copy + center, velocity);
					trashMan.AddDebris(copy - center, -velocity);
				}
			}
		}

		if (addExplosion)
		{
			for (int i = 0; i < 20; i++)
			{
				managedProcess.AddExhaust(silly.GetCenter() + glm::ballRand(0.25f), glm::sphericalRand(5.f), 256);
			}
			for (int i = 0; i < 5; i++)
			{
				glm::vec3 velocity = glm::ballRand(5.f);
				if (glm::length(velocity) < 2.5f)
				{
					velocity *= 2.5f;
				}
				glm::vec3 center = glm::ballRand(0.25f);
				trashMan.AddDebris(silly.GetCenter() + center, velocity);
				trashMan.AddDebris(silly.GetCenter() - center, -velocity);
			}
			addExplosion--;
		}
		trashMan.Update();

		// Tick the animation??
		if (foobarInstance.IsFinished())
		{
			foobar.Start(foobarInstance);
		}
		foobar.Get(foobarInstance).position;
		float playerSpeed = glm::length(playfield.GetVelocity());
		const glm::vec3 playerForward = playfield.GetVelocity();
		/*
		playerMan.Update([&](DebrisManager::Debris& bloke)
			{
				constexpr float MaxAcceleration = 40.f;
				constexpr std::uint16_t MaxAccelerationTime = 256;
				

				bloke.ticksAlive++;
				glm::vec3 delta = playerModel.translation - bloke.transform.position;

				float length = glm::length(delta);
				if (length < 0.25f)
				{
					return true;
				}
				float mySpeed = glm::length(bloke.delta.position);
				glm::vec3 normalized = glm::normalize(delta);
				float ratio = std::min(static_cast<float>(bloke.ticksAlive + 1) / MaxAccelerationTime, 1.f);
				// Maybe this is better, maybe it isn't, I have no idea
				if (bloke.ticksAlive > 1.25 * MaxAccelerationTime)
				{
					normalized *= ratio * MaxAcceleration;
					BasicPhysics::Update(bloke.transform.position, normalized, glm::vec3(0.f));
				}
				else
				{
					glm::vec3 forces = MakePrediction(bloke.transform.position, bloke.delta.position, ratio * MaxAcceleration,
						playerModel.translation, playfield.GetVelocity());

					bloke.transform.rotation = bloke.transform.rotation * bloke.delta.rotation;
					bloke.scale = glm::min(length / 4.f, bloke.scale);
					BasicPhysics::Update(bloke.transform.position, bloke.delta.position, forces);
					BasicPhysics::Clamp(bloke.delta.position, std::max(MaxAcceleration, playerSpeed * 1.25f));
				}
				return glm::any(glm::isnan(bloke.transform.position));
			}
		);*/
		

		groovy.Update();

		managedProcess.Update();

		if (!magnetic.Finished())
		{
			magnetic.Update();
		}

		// End of Tick timekeeping
		auto tickEnd = std::chrono::steady_clock::now();
		long long tickDelta = (tickEnd - tickStart).count();

		maxTickTime = std::max(tickDelta, maxTickTime);
		averageTickTime = gameTickTime.Update(tickDelta / 1000);

		//std::cout << std::chrono::duration<long double, std::chrono::milliseconds::period>(balb - tickStart) << std::endl;
		TimePoint desired{ tickStart.time_since_epoch() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(tickInterval) };
		while (std::chrono::steady_clock::now() < desired) 
		{
			std::this_thread::yield();
		}
		//while (std::chrono::duration<long double, std::chrono::milliseconds::period>(std::chrono::steady_clock::now() - tickStart) < tickInterval);
		
		// TODO: These *should* work, but don't for some inexplicable reason
		//std::this_thread::sleep_for(tickInterval - (balb - tickStart));
		//std::this_thread::sleep_until<std::chrono::steady_clock>(tickStart + tickInterval);
		//std::this_thread::sleep_until<std::chrono::steady_clock>(desired);
		lastStart = tickStart;
		gameTicks++;
		Level::IncrementCurrentTicK();
		if (setPosSemaphore.try_acquire())
		{
			Level::SetPlayerPos(playerModel.translation);
		}
		Level::SetPlayerVel(playfield.GetVelocity());
		//std::cout << std::chrono::duration<long double, std::chrono::milliseconds::period>(std::chrono::steady_clock::now() - balb).count() << std::endl;
		//std::cout << std::chrono::duration<long double, std::chrono::milliseconds::period>(tickInterval).count() << std::endl;
	} while (!shouldClose);
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
	if (state)
	{
		Input::Gamepad::Deactivate();
	}
	shiftHeld = mods & GLFW_MOD_SHIFT;

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
		if (key == GLFW_KEY_GRAVE_ACCENT)
		{
			Input::ToggleUI();
		}
		if (key == GLFW_KEY_V)
		{
			if (magnetic.Finished())
			{
				magnetic.Start({ playfield.GetModel().translation, playfield.GetModel().rotation});
			}
		}
		if (key == GLFW_KEY_U)
		{
			addExplosion++;
		}
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
		if (key == GLFW_KEY_0)
		{
			std::vector<TextureVertex> points{};
			decalVertex.Swap(points);
		}
		if (key == GLFW_KEY_G)
		{
			setPosSemaphore.release();
		}
		if (key == GLFW_KEY_M) cameraPosition.y += 3;
		if (key == GLFW_KEY_N) cameraPosition.y -= 3;
		if (key == GLFW_KEY_ESCAPE) 
		{
			shouldClose = true;
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
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
	glm::vec2 viewPortSize = Window::GetSizeF();
	glm::vec2 sizes((x / viewPortSize.x) * 2.0f - 1.0f, (1.0f - (y / viewPortSize.y)) * 2.0f - 1.0f);

	// Lets have depth = 0.01;
	float depth = 0.01f;
	glm::mat4 projection = Window::GetPerspective(depth, zFar);
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

void mouseButtonFunc(GLFWwindow* window, int button, int action, int status)
{
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
		for (auto& item : Level::Geometry.RayCast(liota))
		{
			if (item->Intersect(liota.initial, liota.delta, rayd) && rayd.depth > 0.f && rayd.depth < rayLength)
			{
				rayLength = rayd.depth;
				point = &(*item);
			}
		}
		// Point displayStartTime has the pointer to the closest element
		//Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.1f, rayLength - 0.5f - 0.2f, 30, 30);
		//pointingCapsule.ReOrient(glm::vec3(0, 0, 90.f));
		//pointingCapsule.ReOrient(cameraOrientation);
		//pointingCapsule.ReCenter(cameraPosition);
		//pointingCapsule.Translate(pointingCapsule.Forward() * (0.3f + rayLength / 2.f));
		//pointingCapsule.Rotate(glm::vec3(0, 0, 90.f));
		//pointingCapsule.ReScale(glm::vec3((rayLength - 0.5f) / 2.f, 0.1f, 0.1f));

		//player.ApplyForces(liota.delta * 5.f, 1.f); // Impulse force
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && userPortion.Contains(Mouse::GetPosition()))
	{
		userPortion.z -= 25;
	}
}

void mouseScrollFunc(GLFWwindow* window, double xDelta, double yDelta)
{
	playerSpeedControl += 0.1f * glm::sign(static_cast<float>(yDelta));
	playerSpeedControl = glm::clamp(playerSpeedControl, 0.f, 1.f);
}

void mouseCursorFunc(GLFWwindow* window, double xPos, double yPos)
{
	float x = static_cast<float>(xPos), y = static_cast<float>(yPos);
	const glm::vec2 oldPos = Mouse::GetPosition();
	glm::ivec2 deviation = glm::ceil(glm::abs(Window::GetHalfF() - oldPos));


	ui_tester = NineSliceGenerate(Window::GetHalfF(), deviation);
	ui_tester_buffer.BufferData(ui_tester, StaticDraw);
	Mouse::SetPosition(x, y);
	targetAngles = glm::vec3(0.f);

	if (Mouse::CheckButton(Mouse::ButtonRight))
	{
		float xDif = x - oldPos.x;
		float yDif = y - oldPos.y;
		if (abs(xDif) > 20)
			xDif = 0;
		if (abs(yDif) > 20)
			yDif = 0;
		// Why 50??
		const float AngleFactor = ANGLE_DELTA * 50.f;
		glm::vec2 rawDeltas = glm::vec2(xDif, yDif) / Window::GetSizeF();
		float yDelta = rawDeltas.x * AngleFactor;
		float zDelta = rawDeltas.y * AngleFactor;
		// TODO: Sensitivity values


		cameraRotation.y += yDelta;
		cameraRotation.x = std::clamp(cameraRotation.x + zDelta, -75.f, 75.f);
		if (Mouse::CheckButton(Mouse::ButtonLeft))
		{
			// TODO: Some clamping to ensure less whackiness

			rawDeltas = glm::radians(rawDeltas * AngleFactor);
			glm::vec3 axis = glm::normalize(rawDeltas.x * glm::vec3(0.f, 1.f, 0.f) + rawDeltas.y * glm::vec3(0.f, 0.f, 1.f));
			float length = glm::length(rawDeltas);
			if (!glm::any(glm::isnan(axis)))
			{
				glm::quat rotation = glm::normalize(glm::angleAxis(length, axis));
				//aboutTheShip = aboutTheShip * rotation;
				// Pretending this doesn't do anything
			}
		}
		//else
		{
			glm::vec2 clamped = glm::clamp((Window::GetHalfF() - glm::vec2(x, y)) / Window::GetHalfF(), glm::vec2(-1.f), glm::vec2(1.f));
			targetAngles.x = clamped.x;
			targetAngles.y = clamped.y;
		}
	}
	else
	{
		buttonToggle = buttonRect.Contains(x, y);
	}
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
	Window::Update(width, height);
	
	cameraUniformBuffer.Generate(DynamicDraw, 2 * sizeof(glm::mat4));
	cameraUniformBuffer.SetBindingPoint(0);
	cameraUniformBuffer.BindUniform();

	glm::mat4 projection = Window::GetPerspective(zNear, zFar);
	cameraUniformBuffer.BufferSubData(projection, sizeof(glm::mat4));

	FilterStruct screenFilters{ MinLinear, MagLinear, BorderClamp, BorderClamp };

	screenSpaceBuffer.Generate(StaticRead, sizeof(glm::mat4));
	screenSpaceBuffer.SetBindingPoint(1);
	screenSpaceBuffer.BindUniform();
	screenSpaceBuffer.BufferSubData(Window::GetOrthogonal());

	// Deferred shading buffers
	// Position
	deferredBuffer.GetColorBuffer<0>().CreateEmpty(Window::GetSize(), InternalFloatRGBA16);
	deferredBuffer.GetColorBuffer<0>().SetFilters();
	// Normal
	deferredBuffer.GetColorBuffer<1>().CreateEmpty(Window::GetSize(), InternalFloatRGBA16);
	deferredBuffer.GetColorBuffer<1>().SetFilters();
	// Color
	deferredBuffer.GetColorBuffer<2>().CreateEmpty(Window::GetSize(), InternalRGBA8);
	deferredBuffer.GetColorBuffer<2>().SetFilters();
	deferredBuffer.GetDepth().CreateEmpty(Window::GetSize(), InternalDepth);
	deferredBuffer.Assemble();

	pointLightBuffer.GetColor().CreateEmpty(Window::GetSize());
	pointLightBuffer.Assemble();
}

void init();

int main(int argc, char** argv)
{
	int error = 0;
	debugFlags.fill(false);

	// Briefly test audio thingy
	if (false)
	{
		ma_result result;
		ma_engine engine;

		result = ma_engine_init(NULL, &engine);
		if (result != MA_SUCCESS) {
			return -1;
		}

		ma_engine_play_sound(&engine, "Audio\\sine_s16_mono_48000.wav", NULL);
		system("pause");
		ma_engine_uninit(&engine);
	}


	windowPointer = nullptr;
	if (!glfwInit())
	{
		LogF("Failed to initialized GLFW.n\n");
		return -1;
	}
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	glfwWindowHint(GLFW_OPENGL_API, GLFW_TRUE);
	glfwWindowHint(GLFW_STEREO, GLFW_FALSE);

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

	// Adjust the window so it is completely on screen
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
	glfwSetScrollCallback(windowPointer, mouseScrollFunc);

	glfwSetJoystickCallback(Input::Gamepad::ControllerStatusCallback);

	// imgui setup
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(windowPointer, true);
	ImGui_ImplOpenGL3_Init();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;


	init();
	window_size_callback(nullptr, Window::Width, Window::Height);

	std::thread ticking{ gameTick };
	ticking.detach();
	glfwSetTime(0);
	while (!glfwWindowShouldClose(windowPointer))
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		idle();
		display();
		glfwSwapBuffers(windowPointer);
		glfwPollEvents();
	}
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	Input::Teardown();
	// TODO: cleanup
	return 0;
}

void Dumber(std::size_t id) {}

void testOBB()
{
	std::cout << "OBB Testing\n";
	std::srand(NULL);
	for (int i = 0; i < 1000; i++)
	{
		glm::vec3 forw = glm::ballRand(1.f), size = glm::ballRand(0.5f);
		OBB tester(glm::degrees(forw), size);
		tester.Translate(glm::ballRand(0.35f));
		for (int x = 0; x < 50; x++)
		{
			glm::vec3 forw2 = glm::ballRand(1.f), size2 = glm::ballRand(0.5f);
			OBB tester2(glm::degrees(forw2), size2);
			tester2.Translate(glm::ballRand(0.75f));
			tester.Overlap(tester2);
		}
	}
	std::cout << "OBB Testing Over\n";
}

void init()
{
	//Input::ControllerStuff();
	//testOBB();
	std::srand(NULL);
	// OpenGL Feature Enabling
	EnableGLFeatures<DepthTesting | FaceCulling | DebugOutput>();
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
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

	Level::Geometry.Resize(glm::vec3(20));

	// TODO: This noise stuff idk man
	//Shader::IncludeInShaderFilesystem("FooBarGamer.gsl", "uniformv.glsl");
	//Shader::IncludeInShaderFilesystem("noise2D.glsl", "noise2D.glsl");

	// SHADER SETUP
	Shader::SetBasePath("Shaders");
	Shader::IncludeInShaderFilesystem("lighting", "lighting.incl");
	Shader::IncludeInShaderFilesystem("camera", "camera.incl");
	Shader::IncludeInShaderFilesystem("frustums", "frustums.incl");
	ExternalShaders::Setup();

	basic.CompileSimple("basic");
	billboardShader.CompileSimple("texture");
	ShaderBank::Get("bulletShader").Compile("color_final", "mesh_final");
	debris.Compile("mesh_final_instance", "mesh_final");
	decalShader.CompileSimple("decal");
	
	engine.CompileSimple("engine");
	fontShader.CompileSimple("font");
	nineSlicer.CompileSimple("ui_nine");
	normalDebris.CompileSimple("mesh_instance");
	pathNodeView.CompileSimple("path_node");
	ship.CompileSimple("mesh_final");
	skyBox.CompileSimple("sky");
	stencilTest.CompileSimple("stencil_");
	trails.CompileSimple("trail");
	uiRect.CompileSimple("ui_rect");
	uiRectTexture.CompileSimple("ui_rect_texture");
	uniform.CompileSimple("uniform");
	ShaderBank::Get("uniform").CompileSimple("uniform");
	vision.CompileSimple("vision");
	widget.CompileSimple("widget");

	ShaderBank::Get("ShieldTexture").Compile(
		"framebuffer", "shield_texture"
	);
	
	ShaderBank::Get("defer").Compile("new_mesh", "deferred");
	ShaderBank::Get("dither").CompileSimple("light_text_dither");
	ShaderBank::Get("expand").Compile("framebuffer", "expand");
	ShaderBank::Get("fullRender").Compile("framebuffer", "full_render");
	ShaderBank::Get("lightVolume").CompileSimple("light_volume");
	ShaderBank::Get("new_mesh").CompileSimple("new_mesh");
	ShaderBank::Get("new_mesh_single").Compile("new_mesh_single", "deferred");
	ShaderBank::Get("uniformInstance").Compile("uniform_instance", "uniform");
	ShaderBank::Get("combinePass").Compile("framebuffer", "combine_pass");
	ShaderBank::Get("light_volume_mesh").CompileSimple("light_volume_mesh");
	ShaderBank::Get("light_volume").CompileSimple("light_volume");
	ShaderBank::Get("Shielding").CompileSimple("shield");

	basic.UniformBlockBinding("Camera", 0);
	billboardShader.UniformBlockBinding("Camera", 0);
	ShaderBank::Get("bulletShader").UniformBlockBinding("Camera", 0);
	debris.UniformBlockBinding("Camera", 0);
	decalShader.UniformBlockBinding("Camera", 0);
	
	engine.UniformBlockBinding("Camera", 0);
	normalDebris.UniformBlockBinding("Camera", 0);
	pathNodeView.UniformBlockBinding("Camera", 0);
	ship.UniformBlockBinding("Camera", 0);
	skyBox.UniformBlockBinding("Camera", 0);
	stencilTest.UniformBlockBinding("Camera", 0);
	trails.UniformBlockBinding("Camera", 0);
	uniform.UniformBlockBinding("Camera", 0);
	vision.UniformBlockBinding("Camera", 0);

	ShaderBank::Get("defer").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("dither").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("fullRender").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("lightVolume").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("new_mesh").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("new_mesh_single").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("uniformInstance").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("combinePass").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("light_volume_mesh").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("light_volume").UniformBlockBinding("Camera", 0);
	ShaderBank::Get("Shields").UniformBlockBinding("Camera", 0);

	nineSlicer.UniformBlockBinding("ScreenSpace", 1);
	uiRect.UniformBlockBinding("ScreenSpace", 1);
	uiRectTexture.UniformBlockBinding("ScreenSpace", 1);
	fontShader.UniformBlockBinding("ScreenSpace", 1);

	debris.UniformBlockBinding("Lighting", 3);
	normalDebris.UniformBlockBinding("Lighting", 3);
	ship.UniformBlockBinding("Lighting", 3);

	ShaderBank::Get("new_mesh").UniformBlockBinding("BlockLighting", 4);
	ShaderBank::Get("fullRender").UniformBlockBinding("BlockLighting", 4);

	// VAO SETUP
	billboardVAO.ArrayFormat<TextureVertex>();
	billboardVAO.ArrayFormatM<glm::mat4>(billboardShader, 1, 1, "Orient");

	fontVAO.ArrayFormat<UIVertex>();

	meshVAO.ArrayFormat<MeshVertex>();

	nineSliced.ArrayFormatOverride<glm::vec4>("rectangle", nineSlicer, 0, 1);

	pathNodeVAO.ArrayFormat<Vertex>(0);
	pathNodeVAO.ArrayFormatOverride<glm::vec3>("Position", pathNodeView, 1, 1);

	plainVAO.ArrayFormat<Vertex>();
	VAOBank::Get("uniform").ArrayFormat<Vertex>();
	VAOBank::Get("meshVertex").ArrayFormat<MeshVertex>();

	VAOBank::Get("engineInstance").ArrayFormatOverride<glm::vec4>(0, 0, 1);
	VAOBank::Get("muscle").ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, 56);
	{
		VAO& ref = VAOBank::Get("uniformInstance");
		ref.ArrayFormat<Vertex>();
		ref.ArrayFormatM<glm::mat4>(ShaderBank::Get("uniformInstance"), 1, 1, "Model");
	}
	{
		VAO& ref = VAOBank::Get("light_volume");
		ref.ArrayFormatOverride<glm::vec4>(0, 0, 1, 0);
		ref.ArrayFormatOverride<glm::vec3>(1, 0, 1, offsetof(LightVolume, color));
		ref.ArrayFormatOverride<glm::vec3>(2, 0, 1, offsetof(LightVolume, constants));
	}
	{
		VAO& ref = VAOBank::Get("light_volume_mesh");
		ref.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, sizeof(MeshVertex));
		ref.ArrayFormatOverride<glm::vec4>(1, 1, 1, 0);
		ref.ArrayFormatOverride<glm::vec3>(2, 1, 1, offsetof(LightVolume, color));
		ref.ArrayFormatOverride<glm::vec3>(3, 1, 1, offsetof(LightVolume, constants));
	}
	{
		VAO& ref = VAOBank::Get("new_mesh_single");
		//ref.ArrayFormat<NormalMeshVertex>();
		ref.ArrayFormatOverride<glm::vec3>(0, 0, 0, offsetof(NormalMeshVertex, position), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(NormalMeshVertex, normal), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(2, 0, 0, offsetof(NormalMeshVertex, tangent), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(3, 0, 0, offsetof(NormalMeshVertex, biTangent), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec2>(4, 0, 0, offsetof(NormalMeshVertex, texture), sizeof(NormalMeshVertex));
	}
	{
		VAO& ref = VAOBank::Get("new_mesh");
		ref.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(NormalMeshVertex, normal), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(2, 0, 0, offsetof(NormalMeshVertex, tangent), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(3, 0, 0, offsetof(NormalMeshVertex, biTangent), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec2>(4, 0, 0, offsetof(NormalMeshVertex, texture), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::mat4>("modelMat", ShaderBank::Get("new_mesh"), 1, 1, 0, sizeof(MeshMatrix));
		//ref.ArrayFormatOverride<glm::mat4>("normalMat", ShaderBank::Get("new_mesh"), 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
	}
	{
		CheckError();
		VAO& ref = VAOBank::Get("simple_mesh_instance");
		ref.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, sizeof(MeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(MeshVertex, normal), sizeof(MeshVertex));
		ref.ArrayFormatOverride<glm::vec2>(2, 0, 0, offsetof(MeshVertex, texture), sizeof(MeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(3, 1, 1, 0, sizeof(glm::vec3));
		//ref.ArrayFormatOverride<glm::mat4>("modelMat", ShaderBank::Get("new_mesh"), 1, 1, 0, sizeof(MeshMatrix));
		//ref.ArrayFormatOverride<glm::mat4>("normalMat", ShaderBank::Get("new_mesh"), 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
	}
	CheckError();
	{
		std::array<glm::vec4, 40*2> lightingArray{ glm::vec4(0.f) };
		std::array<LightVolume, 2> kipper{};
		for (std::size_t i = 0; i < lightingArray.size(); i += 2)
		{
			lightingArray[i] = glm::vec4(glm::ballRand(200.f), 0.f);
			lightingArray[i + 1] = glm::vec4(glm::abs(glm::ballRand(1.f)), 0.f);
			//std::cout << lightingArray[i] << ":" << lightingArray[i + 1] << '\n';
			constantLights.push_back({lightingArray[i], glm::vec3(lightingArray[i + 1]), glm::vec3(1.f, 1.f / 30.f, 0.002f)});
			constantLights.back().position.w = 60.f;
		}
		globalLighting.BufferData(lightingArray);
		globalLighting.SetBindingPoint(4);
		globalLighting.BindUniform();
		Bank<ArrayBuffer>::Get("light_volume_mesh").BufferData(kipper);
		Bank<ArrayBuffer>::Get("dummy").BufferData(std::array<glm::vec3, 4>());
		Bank<ArrayBuffer>::Get("dummy2").BufferData(std::array<glm::vec3, 10>());
	}

	texturedVAO.ArrayFormat<TextureVertex>();

	colorVAO.ArrayFormat<ColoredVertex>();

	lightingBuffer.Generate(DynamicDraw, sizeof(glm::vec4) * 2);
	std::array<glm::vec4, 2> locals{ glm::vec4(glm::abs(glm::ballRand(1.f)), 0.f), glm::vec4(0.15f, 1.f, 0.15f, 0.f) };
	lightingBuffer.BufferSubData(locals, 0);
	lightingBuffer.SetBindingPoint(3);
	lightingBuffer.BindUniform();

	// TEXTURE SETUP
	// These two textures from https://opengameart.org/content/stylized-mossy-stone-pbr-texture-set, do a better credit
	Texture::SetBasePath("Textures");

	ditherTexture.Load(Dummy::dither16, InternalRed, FormatRed, DataUnsignedByte);
	ditherTexture.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	hatching.Load("hatching.png");
	hatching.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	//normalMap.Load("bear_nm.png");
	normalMap.Load("normal.png");
	normalMap.SetFilters(LinearLinear, MagLinear, MirroredRepeat, MirroredRepeat);
	normalMap.SetAnisotropy(16.f);

	//texture.Load("laserA.png");
	texture.Load("laserC.png"); // Temp switching to a properly square decal
	texture.SetFilters(LinearLinear, MagLinear, BorderClamp, BorderClamp);

	wallTexture.Load("flowed.png");
	wallTexture.SetFilters(LinearLinear, MagNearest, Repeat, Repeat);

	buttonB.CreateEmptyWithFilters(100, 100, InternalRGBA, {}, glm::vec4(0, 1, 1, 1));
	buttonA.CreateEmptyWithFilters(100, 100, InternalRGBA, {}, glm::vec4(1, 0.5, 1, 1));

	nineSlice.Load("9slice.png");
	nineSlice.SetFilters();
	Bank<Texture2D>::Get("flma").Load("depth.png");
	Bank<Texture2D>::Get("flma").SetFilters();

	// TODO: Use glm::noise::perlin

	Bank<ArrayBuffer>::Get("dummyEngine").BufferData(std::array<unsigned int, 36>{});

	decals.Generate();
	stickBuffer.BufferData(Dummy::stick);
	solidCubeIndex.BufferData(Cube::GetTriangleIndex());

	glLineWidth(100.f);
	Bank<ArrayBuffer>::Get("plainCube").BufferData(Cube::GetPoints());

	Bank<Texture2D>::Get("blankTexture").CreateEmpty(1, 1, InternalRGBA8, glm::vec4(1.f));
	Bank<Texture2D>::Get("blankTexture").SetFilters();

	//std::array<glm::vec3, 5> funnys = { {glm::vec3(0.25), glm::vec3(0.5), glm::vec3(2.5, 5, 3), glm::vec3(5, 2, 0), glm::vec3(-5, 0, -3) } };
	//pathNodePositions.BufferData(funnys);

	{
		ShaderBank::Get("gleep").CompileCompute("light_cull");
		Shader& shader = ShaderBank::Get("computation");
		shader.CompileCompute("compute_frustums");
		ShaderStorageBuffer locality;
		locality.BufferData(std::array<std::uint32_t, 256>{0});

		locality.BindBuffer();
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, locality.GetBuffer());

		auto nextMult = [](auto a, auto b) {return glm::ceil(a / b) * b; };

		const int TileSize = 16;
		const float TileSizeF = static_cast<float>(TileSize);
		glm::uvec2 windowSize = nextMult(Window::GetSizeF(), TileSizeF); // glm::ceil(Window::GetSizeF() / TileSizeF)* TileSizeF;

		// Moving past the sample
		shader.SetActiveShader();
		shader.SetInt("Width", windowSize.x);
		shader.SetInt("Height", windowSize.y);
		shader.DispatchCompute(257);
	}


	// RAY SETUP
	std::array<glm::vec3, 20> rays = {};
	rays.fill(glm::vec3(0));
	rayBuffer.BufferData(rays);

	heWhoSleeps.Setup();
	heWhoSleeps.openStyle = Door::Type::Triangle;
	heWhoSleeps.openState = Door::Closed;
	heWhoSleeps.openTicks = heWhoSleeps.closingDuration;
	//heWhoSleeps.openState = Door::Opening;
	//heWhoSleeps.openTicks = 125;
	heWhoSleeps.model.scale = glm::vec3(17.5f);
	heWhoSleeps.model.rotation = ForwardDir(glm::vec3(-1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
	heWhoSleeps.model.rotation = QuatIdentity();
	heWhoSleeps.model.translation = glm::vec3(97.244f, 17.102f, 0);
	std::vector<glm::vec3> fdso;
	auto sd = heWhoSleeps.GetTris();
	for (const Triangle& p : sd)
	{
		for (const glm::vec3& sdf : p.GetPointArray())
			fdso.push_back(sdf);
	}
	rayBuffer.BufferData(fdso);

	// This sucks
	// TODO: Put this in Geometry, or something, I don't know
	std::array<TextureVertex, 36> textVert{};
	for (std::size_t i = 0; i < 36; i++)
	{
		textVert[i].position = Cube::GetUVPoints()[i].position;
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
	decals.Generate();

	auto verts = Planes::GetUVPoints();
	for (auto& point : verts)
	{
		point.position = glm::mat3(glm::eulerAngleZY(glm::radians(90.f), glm::radians(-90.f))) * point.position;
		//point.position += glm::vec3(0, 1.f, 0);
	}
	billboardBuffer.BufferData(verts);
	constexpr int followsize = 10;
	//followers.ReserveSize(followsize);
	for (int i = 0; i < followsize; i++)
	{
		glm::vec2 base = glm::diskRand(20.f);
		//followers.emplace_back(glm::vec3(base.x, 2.5, base.y));
		PathFollower fus(glm::vec3(base.x, 2.5, base.y));
		//followers.Insert(fus, fus.GetAABB());
	}
	PathFollower sc(glm::vec3(-100, 2.5, -100));
	PathFollower sc2(glm::vec3(-10, 2.5, -10));
	//followers.Insert(sc, sc.GetAABB());
	//followers.Insert(sc2, sc2.GetAABB());

	// Cube map shenanigans
	{
		// From Here https://opengameart.org/content/space-skybox-1 under CC0 Public Domain License
		sky.Generate(std::to_array<std::string>({"skybox/space_ft.png", "skybox/space_bk.png", "skybox/space_up.png", 
			"skybox/space_dn.png", "skybox/space_rt.png", "skybox/space_lf.png"}));
	}

	tickTockMan.Init();
	Parallel::SetStatus(true);
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

	// =============================================================
	// Pathfinding stuff

	{
		QuickTimer _timer("Node Connections");


		// TODO: Investigate with better optimized kdtree stuff

		// TODO: hash pair collide thingy so nodes don't have to recalculate the raycast

		{
			//QUICKTIMER("Thing B");
			//std::size_t countes = 0;
			for (std::size_t i = 0; i < Level::AllNodes().size(); i++)
			{
				for (std::size_t j = i + 1; j < Level::AllNodes().size(); j++)
				{
					PathNode::addNeighbor(Level::AllNodes()[i], Level::AllNodes()[j],
						[&](const PathNodePtr& A, const PathNodePtr& B)
						{
							glm::vec3 a = A->GetPosition(), b = B->GetPosition();
							float delta = glm::length(a - b);
							if (delta > 5.f) // TODO: Constant
								return false;
							Ray liota(a, b - a);
							//countes++;
							auto temps = Level::Geometry.RayCast(liota);
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
			//std::cout << "Count: " << countes << std::endl;
		}

		// TODO: Second order check to remove connections that are "superfluous", ie really similar in an unhelpful manner
		std::erase_if(Level::AllNodes(), [](const PathNodePtr& A) {return A->neighbors().size() == 0; });
		//Parallel::erase_if(std::execution::par_unseq, Level::AllNodes, [](const PathNodePtr& A) {return A->neighbors().size() == 0; });
		for (std::size_t i = 0; i < Level::AllNodes().size(); i++)
		{
			auto& local = Level::AllNodes()[i];
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

	std::vector<glm::vec3> nodePoints;
	std::vector<Triangle> nodeTri;
	std::vector<glm::vec3> painterly;
	int onlyFirst = 0;
	{
		QUICKTIMER("Model Loading");
		std::vector<glm::vec3> badBoxes;
		guyMeshData = OBJReader::MeshThingy<NormalMeshVertex>("Models\\bloke6.obj", {}, 
			[&](auto& c)
			{ 
				std::ranges::transform(c, std::back_inserter(badBoxes), [](NormalMeshVertex b) -> glm::vec3 {return b.position; });
			}
		);
		ClockBrain::Collision = OBB::MakeOBB(badBoxes);
		playerMesh = OBJReader::MeshThingy<MeshVertex>("Models\\Player.glb", {}, 
			[&](auto& c) -> void
			{
				if (onlyFirst++)
					return;
				std::ranges::transform(c, std::back_inserter(painterly), [](MeshVertex b) -> glm::vec3 {return b.position; });
			}
		);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		playerMesh2 = OBJReader::MeshThingy<NormalMeshVertex>("Models\\Player.glb");
		Player::Box = OBB::MakeOBB(painterly);
		Player::Box.Scale(0.5f);
		bulletMesh = OBJReader::MeshThingy<ColoredVertex>("Models\\Projectiles.glb",
			{},
			[&](auto& c)
			{
				std::vector<glm::vec3> pain{ c.size() };
				std::ranges::transform(c, std::back_inserter(pain), [](ColoredVertex b) -> glm::vec3 {return b.position; });
				Bullet::Collision = OBB::MakeOBB(pain);
			}
		);
		//geometry = OBJReader::MeshThingy("Models\\LevelMaybe.glb",
		//levelGeometry = OBJReader::MeshThingy<NormalMeshVertex>("Models\\LevelMaybe2.glb",
		levelGeometry = OBJReader::MeshThingy<NormalMeshVertex>("Models\\mothership.glb",
		//levelGeometry = OBJReader::MeshThingy<NormalMeshVertex>("Models\\big_box.obj",
			[&](const auto& c)
			{
				if (c.size() >= 3)
				{
					Triangle local(c[0], c[1], c[2]);
					nodeTri.push_back(local);
					Level::AddTri(local);
				}
			}
		);

		Bank<OBB>::Get("NoGoZone") = OBB(AABB(glm::vec3(30.f)));
		//levelGeometry.rawIndirect[0].vertexCount = levelGeometry.index.GetElementCount();
		//levelGeometry.indirect.BufferData(levelGeometry.rawIndirect[0]);
		std::cout << levelGeometry.index.GetElementCount() << '\n';
		std::cout << levelGeometry.vertex.GetElementCount() << '\n';
	}
	Bank<ArrayBuffer>::Get("dummyInstance").BufferData(std::to_array<MeshMatrix>({ {glm::mat4(1.f), glm::mat4(1.f)} }));

	ShieldGenerator::Setup();

	BSP& bp = Bank<BSP>::Get("Fellas");
	{
		QUICKTIMER("BSP Tree");
		bp.GenerateBSP(nodeTri);
	}
	
	for (int i = 0; i < 10; i++)
	{
		management.Make().Init(i > 5 ? glm::vec3(0.f, 60.f, 0.f) : glm::vec3(0.f, -60.f, 0.f));
	}

	Level::GetTriangleTree().UpdateStructure();
	nodePoints.clear();
	/*
	std::size_t remo = 0;
	int bouncy = 0;
	int increment = 25;
	for (int i = -bouncy; i <= bouncy; i += increment)
	{
		for (int j = -bouncy; j <= bouncy; j += increment)
		{
			for (int k = -bouncy; k <= bouncy; k += increment)
			{
				glm::vec3 important(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
				if (bp.TestPoint(important))
				{
					nodePoints.push_back(important);
					Level::AllNodes().push_back(PathNode::MakeNode(important));
				}
				else
				{
					remo++;
				}
			}
		}
	}
	std::cout << "Culled Nodes:" << remo << '\n';

	std::vector<glm::vec3> littleTrolling{};
	NavMesh goober("oops");
	if (!goober.Load("oops"))
	{
		goober.Generate(std::span(nodePoints),
			[](const NavMesh::Node& A, const NavMesh::Node& B)
			{
				glm::vec3 a = A.position, b = B.position;
				float delta = glm::length(a - b);
				if (delta > 36.f) // TODO: Constant
					return false;
				Ray liota(a, b - a);
				auto temps = Level::GetTriangleTree().RayCast(liota);
				if (temps.size() == 0)
				{
					return true;
				}
				for (auto& temp : temps)
				{
					RayCollision fumop{};
					if (temp->RayCast(liota, fumop) && fumop.depth > 0 && fumop.depth < delta)
					{
						return false;
					}
				}
				return true;
			});
		goober.Export();
	}
	std::size_t itemOffset = 123;
	for (glm::vec3 point : goober.AStar(0, static_cast<NavMesh::IndexType>(goober.size() / 2), 
		[](const NavMesh::Node& a, const NavMesh::Node& b) -> float
		{
			return a.distance(b); 
		}
	))
	{
		littleTrolling.push_back(point);
		std::cout << point << '\n';
	}*/

	{
		QUICKTIMER("AABB Stress test");
		std::size_t succeed = 0, fails = 0;
		Level::GetTriangleTree().for_each(
			[&](auto& ref) 
			{
				const glm::vec3 start = ref.GetCenter() + ref.GetNormal() * 2.f;
				const AABB box = ref.GetAABB();
				for (auto i = 0; i < 100; i++)
				{
					const glm::vec3 direction = glm::sphericalRand(1.f);
					Ray liota(start, direction);
					if (box.FastIntersect(liota) == box.Intersect(start, direction))
					{
						succeed++;
					}
					else
					{
						Log(std::boolalpha << box.FastIntersect(liota) << ":" << box.Intersect(start, direction));
						fails++;
					}

				}
				return false; 
			}
		);
		Log(std::format("Pass {} : Fail {}", succeed, fails));
	}

	std::cout << "Big Node Size: " << Level::AllNodes().size() << '\n';
	{
		QUICKTIMER("KdTree Generation");
		Level::Tree = kdTree<PathNodePtr>::Generate(Level::AllNodes());
	}
	if (false)
	{
		QUICKTIMER("Node Connections");

		/*
		std::for_each(Level::AllNodes.begin(), Level::AllNodes.end(),
			[](PathNodePtr& current)
			{
				for (auto& inner : Level::Tree.neighborsInRange(current->GetPos(), 20.f))
				{
					PathNode::addNeighborUnconditional(current, inner);
				}
			}
		);
		*/
		std::vector<glm::vec3> foolish;
		for (std::size_t i = 0; i < Level::AllNodes().size(); i++)
		{
			for (std::size_t j = i + 1; j < Level::AllNodes().size(); j++)
			{
				PathNode::addNeighbor(Level::AllNodes()[i], Level::AllNodes()[j],
					[](const PathNodePtr& A, const PathNodePtr& B)
					{
						glm::vec3 a = A->GetPosition(), b = B->GetPosition();
						float delta = glm::length(a - b);
						if (delta > 20.f) // TODO: Constant
							return false;
						Ray liota(a, b - a);
						auto temps = Level::GetTriangleTree().RayCast(liota);
						if (temps.size() == 0)
						{
							return true;
						}
						for (auto& temp : temps)
						{
							RayCollision fumop{};
							if (temp->RayCast(liota, fumop) && fumop.depth > 0 && fumop.depth < delta)
							{
								return false;
							}
						}
						return true;
					}
				);
			}
		}
	}

	if (Level::AllNodes().size() > 0)
	{
		QUICKTIMER("kdTree");
		const auto& first = Level::AllNodes().front();
		PathNodePtr pint = nullptr;
		float dist = INFINITY;
		Level::Tree.nearestNeighbor(first->GetPos());
	}
	if (Level::AllNodes().size() > 0) 
	{
		QUICKTIMER("Linear");
		const auto& first = Level::AllNodes().front();
		PathNodePtr pint = nullptr;
		float dist = INFINITY;
		for (const auto& b : Level::AllNodes())
		{
			if (b == first)
				continue;
			if (glm::distance(first->GetPos(), b->GetPos()) < dist)
			{
				dist = glm::distance(first->GetPos(), b->GetPos());
				pint = b;
			}
		}
	}
	std::vector<glm::vec3> boxingDay{};
	// Don't need this crap
	std::size_t lineCount = 0;
	
	
	//for (const auto& p : goober)
	{
		//boxingDay.push_back(p.position);
		/*
		for (const NavMesh::IndexType& weak : p.connections)
		{
			if (weak < boxingDay.size())
			{
				continue;
			}
			const auto& weaker = goober.begin() + weak;
			littleTrolling.push_back(p.position);
			littleTrolling.push_back(weaker->position);
			lineCount++;
		}*/
	}
	std::cout << "Total Edges: " << lineCount / 2 << '\n';
	//Bank<ArrayBuffer>::Get("nodePositions").BufferData(boxingDay, StaticDraw);
	//Bank<ArrayBuffer>::Get("nodeLinePositions").BufferData(littleTrolling, StaticDraw);

	// =============================================================

	{
		QuickTimer _tim{ "Sphere/Capsule Generation" };
		Sphere::GenerateMesh(sphereBuffer, sphereIndicies, 100, 100);
		Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.75f, 3.25f, 30, 30);
	}

	{
		QUICKTIMER("Level Triangle Allocation");
		std::vector<glm::vec3> matrixif;
		for (const auto& tri : Level::GetTriangleTree())
		{
			for (const auto& point : tri.GetPointArray())
			{
				matrixif.push_back(point);
			}
		}
		
		levelOutline.BufferData(matrixif);
	}

	//MeshThingy("Models\\Debris.obj");

	bulletVAO.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0);
	bulletVAO.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(ColoredVertex, color));
	bulletVAO.ArrayFormatOverride<glm::mat4>("modelMat", ShaderBank::Get("bulletShader"), 1, 1, 0, sizeof(glm::mat4));

	// TODO: Figure out why std::move(readobj) has the wrong number of elements
	//std::cout << satelitePairs.size() << ":\n";
	Font::SetFontDirectory("Fonts");
	
	DebrisManager::LoadResources();
	trashMan.Init();
	Satelite::LoadResources();
	// Doing this should not change anything, why does it fix things
	//trashMan.AddDebris(glm::vec3(5, 5, 0), World::Zero);

	// Awkward syntax :(

	{
		QUICKTIMER("Font Loading");
		ASCIIFont::LoadFont(fonter, "CommitMono-400-Regular.ttf", 25.f, 2, 2);
	}

	// TODO: proper fill of the relevant offset so there's no weird banding
	leftCircle.Fill({ playfield.GetModel().translation, glm::vec3(0.f) });
	rightCircle.Fill({playfield.GetModel().translation, glm::vec3(0.f)});
	leftBuffer.BufferData(leftCircle.Get());
	rightBuffer.BufferData(rightCircle.Get());
	stickIndicies.BufferData(Dummy::stickDex, StaticDraw);

	cubeOutlineIndex.BufferData(Cube::GetLineIndex());

	dumbBox.ReCenter(glm::vec3(0, 1.f, -2));
	dumbBox.Scale(glm::vec3(1.f));
	dumbBox.Rotate(glm::vec3(0, -90, 0));
	
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
		nineSliced.Bind();
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

	playfield.sat = &groovy;
	Input::Setup();
	glClearColor(0.f, 0.f, 0.f, 0.f);
	Log("End of Init");
}