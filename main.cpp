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
#pragma warning (push)
#pragma warning (disable : 6031 6011 33010 28182 26819)
#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_impl_glfw.h"
#pragma warning (pop)
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
#include "Test.h"

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

UniformBuffer cameraUniformBuffer, screenSpaceBuffer;

// Textures
Texture2D ditherTexture, hatching, normalMap, texture, wallTexture;
Texture2D buttonA, buttonB, nineSlice;
CubeMap sky;

// Vertex Array Objects
VAO fontVAO, pathNodeVAO, meshVAO, plainVAO, texturedVAO;
VAO nineSliced;
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

float zNear = 0.1f, zFar = 1000.f;

bool buttonToggle = false;
ScreenRect buttonRect{ 540, 200, 100, 100 }, userPortion(0, 800, 1000, 200);
Button help(buttonRect, [](std::size_t i) {std::cout << idleFrameCounter << ":" << i << std::endl; });

static bool featureToggle = false;
static std::chrono::nanoseconds idleTime, displayTime;

struct LightVolume
{
	// Position position.w means point light, being the radius
	// Negative position.w means cone, with the absolute value of it being the height
	// 0 position.w means directed light
	glm::vec4 position{ glm::vec3(0.f), 10.f };
	glm::vec4 color{ 1.f };
	glm::vec4 constants{1.f, 0.025f, 0.f, 0.f};
	glm::vec4 direction{0.f};
};

// TODO: Semaphore version of buffersync
BufferSync<std::vector<TextureVertex>> decalVertex;

std::array<ScreenRect, 9> ui_tester;
ArrayBuffer ui_tester_buffer;

std::vector<AABB> dynamicTreeBoxes;
using namespace Input;

static bool windowShouldClose = false;

using TimePoint = std::chrono::steady_clock::time_point;
using TimeDelta = std::chrono::nanoseconds;
static std::size_t gameTicks = 0;

SimpleAnimation foobar{ {glm::vec3(-0.025, 0, 0)}, 32, Easing::Quintic,
						{glm::vec3(-0.25, 0, 0)}, 80, Easing::Linear };
AnimationInstance foobarInstance;
ExhaustManager managedProcess;

Player playfield(glm::vec3(0.f, 50.f, 0.f));
float playerSpeedControl = 0.1f;
Input::Keyboard boardState; 
// TODO: Proper start/reset value
glm::quat aboutTheShip(0.f, 0.f, 0.f, 1.f);

Satelite groovy{ glm::vec3(10.f, 10.f, 0) };
bool shiftHeld;
std::atomic_uchar addExplosion;

DebrisManager trashMan;

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

glm::vec4 testCameraPos(-30.f, 15.f, 0.f, 60.f);
BufferSync<std::vector<LightVolume>> drawingVolumes;
std::vector<LightVolume> constantLights;

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

	const glm::vec3 modelForward = playerModel.rotation * glm::vec3(1.f, 0.f, 0.f);
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
Framebuffer renderTarget;
BufferSync<std::vector<glm::vec3>> shieldPos;
using ShaderStorage = Bank<ShaderStorageBuffer>;

static const float gridResolution = 16;
static int numTiles = 0;
static glm::uvec2 tileDimension;
static Framebuffer<1, Depth> earlyDepth;
static constexpr float EarlyDepthRatio = 1;

static std::vector<GLuint> glQueries;
std::mutex bulletMutex;

constexpr std::size_t dustDimension = 20;
constexpr std::size_t dustCount = dustDimension * dustDimension * dustDimension;

const glm::vec3 flashLightColor = glm::vec3(148.f, 252.f, 255.f) / 255.f;

void BindDrawFramebuffer()
{
	renderTarget.BindDraw();
	Window::Viewport();
}

void display()
{
	GLuint currentRenderQuery = 0;
	glGenQueries(1, &currentRenderQuery);
	glBeginQuery(GL_TIME_ELAPSED, currentRenderQuery);

	auto displayStartTime = std::chrono::high_resolution_clock::now();

	EnableGLFeatures<DepthTesting | FaceCulling>();
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
	}
	BindDrawFramebuffer();
	glClearDepth(0);
	ClearFramebuffer<ColorBuffer | DepthBuffer | StencilBuffer>();

	const Model playerModel(playfield.GetModel());

	// Camera matrix
	const glm::mat3 axes(playerModel.rotation);
	const glm::vec3 velocity = playfield.GetVelocity();
	
	std::pair<glm::vec3, glm::vec3> cameraPair = CalculateCameraPositionDir(playerModel);
	const glm::vec3 localCamera = cameraPair.first;
	const glm::vec3 cameraForward = cameraPair.second;

	const glm::mat4 view = glm::lookAt(localCamera, GetCameraFocus(playerModel, velocity), axes[1]);
	cameraUniformBuffer.BufferSubData(view, 0);
	Frustum frustum(localCamera, ForwardDir(cameraForward, axes[1]), glm::vec2(zNear, zFar));
	CheckError();
	glEnable(GL_CULL_FACE);
	//glDepthFunc(GL_LEQUAL);
	glDepthFunc(GL_GEQUAL);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	Shader& uniform = ShaderBank::Retrieve("uniform");
	if (debugFlags[DEBUG_PATH])
	{
		EnableGLFeatures<Blending>();
		//DisableDepthBufferWrite();
		Shader& pathNodeView = ShaderBank::Retrieve("pathNodeView");
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
	drawingVolumes.ExclusiveOperation(
		[&](std::vector<LightVolume>& data)
		{
			// The players 'torch'
			// Has to be like this so it isn't duplicated
			LightVolume greeblies;
			// 100 is the length of the cone
			constexpr float FlashLightHeight = 100.f;
			constexpr float FlashLightRadius = 50.f;
			greeblies.position = glm::vec4(playerModel.translation, -FlashLightHeight);
			greeblies.color = glm::vec4(flashLightColor, 1.f);
			greeblies.constants = glm::vec4(1.f, 1.f / 20.f, 1.f / 2000.f, 1.f);
			greeblies.direction = glm::vec4(axes[0], FlashLightRadius);
			data.push_back(greeblies);
			
			ShaderStorage::Retrieve("LightBlockOriginal").BufferData(data);
			// TODO: Work around this hacky thing, I don't like having to use double the memory for lights
			auto& buffer = ShaderStorage::Retrieve("LightBlock");
			std::vector<LightVolume> grouper;
			std::ranges::copy(
				data | std::views::transform(
					[&](const LightVolume& v)
					{
						glm::vec3 transformed = view * glm::vec4(glm::xyz(v.position), 1.f);
						glm::vec3 transformed2 = view * glm::vec4(glm::xyz(v.direction), 0.f);
						return LightVolume{ glm::vec4(transformed, v.position.w),
								v.color, v.constants, glm::vec4(transformed2, v.direction.w)};
					}
				), std::back_inserter(grouper));

			buffer.BufferData(grouper);
			ShaderStorage::Retrieve("LightGrid2").BufferSubData(static_cast<std::uint32_t>(data.size()), 0);
			ShaderStorage::Retrieve("LightGrid2").BufferSubData<std::uint32_t>(0, sizeof(std::uint32_t));

			data.pop_back();
		}
	);
	// Compute Shaders
	{
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		Shader& cullLights = ShaderBank::Retrieve("lightCulling");
		cullLights.SetActiveShader();
		cullLights.SetTextureUnit("DepthBuffer", earlyDepth.GetDepth(), 1);
		cullLights.DispatchCompute(tileDimension.x, tileDimension.y);
		Shader& computation = ShaderBank::Retrieve("debrisCompute");
		auto& rawDebris = ShaderStorage::Retrieve("RawDebris");
		auto& transformedOut = ShaderStorage::Retrieve("DrawDebris");
		auto& indirectOut = ShaderStorage::Retrieve("DebrisIndirect");

		computation.SetActiveShader();
		rawDebris.BindBufferBase(0);
		transformedOut.BindBufferBase(1);
		indirectOut.BindBufferBase(2);
		computation.SetFloat("zFar", zFar);
		computation.SetVec3("cameraForward", cameraForward);
		computation.SetVec3("cameraPos", localCamera);
		computation.SetVec3("cameraVelocity", velocity);
		computation.DispatchCompute(dustDimension, dustDimension, dustDimension);

		// END OF COMPUTE SECTION
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);



		auto& outputBuffer = BufferBank::Retrieve("DrawDebris");
		auto& outputIndirect = Bank<DrawIndirectBuffer>::Retrieve("DebrisIndirect");
		glCopyNamedBufferSubData(transformedOut.GetBuffer(), outputBuffer.GetBuffer(), 0, 0, transformedOut.Size());
		glCopyNamedBufferSubData(indirectOut.GetBuffer(), outputIndirect.GetBuffer(), 0, 0, sizeof(unsigned int) * 4);
	}
	// Actual drawing based on the lighting stuff

	Shader& interzone = ShaderBank::Retrieve("forwardPlusMulti");
	VAO& outerzone = VAOBank::Retrieve("forwardPlusMulti");
	interzone.SetActiveShader();
	levelGeometry.Bind(outerzone);
	outerzone.BindArrayBuffer(levelGeometry.vertex, 0);
	interzone.SetVec3("shapeColor", glm::vec3(1.0, 1.0, 1.0));
	interzone.SetVec3("CameraPos", localCamera);
	outerzone.BindArrayBuffer(Bank<ArrayBuffer>::Retrieve("dummyInstance"), 1);

	ShaderStorage::Retrieve("LightIndicies").BindBufferBase(6);
	// Only need one per tile
	//ShaderStorage::Get("LightGrid").BindBufferBase(7);
	//ShaderStorage::Get("LightBlock").BindBufferBase(8);
	interzone.DrawElements<DrawType::Triangle>(levelGeometry.indirect);



	management.Draw(guyMeshData, outerzone, interzone);
	//bobert.Draw();
	
	auto& buf = BufferBank::Get("player");
	auto meshs2 = playerModel;
	meshs2.scale *= 0.5f;
	auto meshs = meshs2.GetMatrixPair();
	buf.BufferData(std::to_array({ meshs.model, meshs.normal }));
	outerzone.BindArrayBuffer(buf, 1);
	playfield.Draw(interzone, outerzone, playerMesh2, playerModel);
	
	{
		// *Needs* to have the copying between the buffers completed by this point
		glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
		Shader& local = ShaderBank::Retrieve("dust");
		local.SetActiveShader();
		VAO& vao = VAOBank::Get("bigscrem");
		vao.Bind();
		vao.BindArrayBuffer(BufferBank::Retrieve("DrawDebris"), 0);
		local.SetVec3("shapeColor", glm::vec3(0.9f));
		local.DrawArrayIndirect<DrawType::TriangleStrip>(Bank<DrawIndirectBuffer>::Retrieve("DebrisIndirect"));
	}

	if (debugFlags[CHECK_UVS])
	{
		Shader& sahder = ShaderBank::Get("visualize");
		sahder.SetActiveShader();
		//sahder.SetVec2("ScreenSize", Window::GetSizeF());
		//sahder.SetInt("TileSize", static_cast<int>(gridResolution));
		//sahder.SetUVec2("tileDimension", tileDimension);
		static int thresholdAmount = 10;
		ImGui::Begin("Light Threshold");
		ImGui::SliderInt("Threshold", &thresholdAmount, 0, 100);
		ImGui::End();
		sahder.SetInt("maxLight", thresholdAmount);
		//FeatureFlagPush<DepthTesting | FaceCulling, false> flagger;
		DisablePushFlags(DepthTesting | FaceCulling);
		sahder.DrawArray<DrawType::TriangleStrip>(4);
	}


	// Copy the current depth buffer status to the early depth buffer for next frame
	glBindFramebuffer(GL_READ_FRAMEBUFFER, renderTarget.GetFrameBuffer());
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, earlyDepth.GetFrameBuffer());
	glm::ivec2 dimension = Window::GetSize();
	glm::ivec2 depthSize = earlyDepth.GetDepth().GetSize();
	glBlitFramebuffer(0, 0, dimension.x, dimension.y, 0, 0, depthSize.x, depthSize.y, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderTarget.GetFrameBuffer());

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

	CheckError();

	Model defaults(playerModel);
	ShaderBank::Get("ship").SetActiveShader();
	meshVAO.Bind();
	defaults.translation = glm::vec3(10, 10, 0);
	defaults.rotation = glm::quat(0.f, 0.f, 0.f, 1.f);
	defaults.scale = glm::vec3(0.5f);
	ShaderBank::Get("ship").SetMat4("modelMat", defaults.GetModelMatrix());
	ShaderBank::Get("ship").SetMat4("normalMat", defaults.GetNormalMatrix());
	groovy.Draw(ShaderBank::Get("ship"));

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	EnableGLFeatures<Blending>();
	DisableDepthBufferWrite();
	// TODO: Maybe look into this https://www.opengl.org/archives/resources/code/samples/sig99/advanced99/notes/node20.html
	ShaderBank::Get("decalShader").SetActiveShader();
	texturedVAO.Bind();
	texturedVAO.BindArrayBuffer(decals);
	ShaderBank::Get("decalShader").SetTextureUnit("textureIn", texture, 0);
	ShaderBank::Get("decalShader").DrawArray<DrawType::Triangle>(decals);
	EnableDepthBufferWrite();
	DisableGLFeatures<Blending>();
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//meshVAO.BindArrayBuffer(guyBuffer2);

	trashMan.Draw(ShaderBank::Retrieve("debris"));
	glLineWidth(1.f);

	ShaderBank::Get("basic").SetActiveShader();
	meshVAO.Bind();
	meshVAO.BindArrayBuffer(sphereBuffer);
	sphereIndicies.BindBuffer();
	ShaderBank::Get("basic").SetVec4("Color", glm::vec4(2.f, 204.f, 254.f, 250.f) / 255.f);
	ShaderBank::Get("basic").SetMat4("Model", magnetic.GetMatrix(playerModel.translation));
	ShaderBank::Get("basic").DrawElements<DrawType::Lines>(sphereIndicies);
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

	{
		DisablePushFlags(FaceCulling);
		EnablePushFlags(Blending);
		Shader& trails = ShaderBank::Retrieve("trail");
		trails.SetActiveShader();
		colorVAO.Bind();
		trails.SetVec3("Color", glm::vec3(2.f, 204.f, 254.f) / 255.f);
		colorVAO.BindArrayBuffer(leftBuffer);
		trails.DrawArray<DrawType::TriangleStrip>(leftBuffer);
		colorVAO.BindArrayBuffer(rightBuffer);
		trails.DrawArray<DrawType::TriangleStrip>(rightBuffer);
	}

	//tickTockMan.Draw(guyMeshData, VAOBank::Get("new_mesh"), ShaderBank::Get("new_mesh"));
	//debris.SetActiveShader();
	//management.Draw(guyMeshData, meshVAO, debris);

	{
		Shader& engine = ShaderBank::Retrieve("engine");
		engine.SetActiveShader();
		VAOBank::Get("engineInstance").Bind();
		VAOBank::Get("engineInstance").BindArrayBuffer(exhaustBuffer);
		engine.SetUnsignedInt("Time", static_cast<unsigned int>(gameTicks & std::numeric_limits<unsigned int>::max()));
		engine.SetUnsignedInt("Period", 150);
		engine.DrawArrayInstanced<DrawType::Triangle>(Bank<ArrayBuffer>::Get("dummyEngine"), exhaustBuffer);
	}
	//EnableGLFeatures<DepthTesting>();
	// 
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
			glLineWidth(1.f);
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
	Shader& stencilTest =  ShaderBank::Retrieve("stencilTest");
	stencilTest.SetActiveShader();
	stencilTest.SetMat4("Model", sphereModel.GetModelMatrix());
	meshVAO.BindArrayBuffer(sphereBuffer);
	//stencilTest.DrawElements<DrawType::Triangle>(sphereIndicies);
	
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
	{
		Shader& uiRect = ShaderBank::Retrieve("uiRect");
		uiRect.SetActiveShader();
		uiRect.SetVec4("color", glm::vec4(0, 0, 0, 0.8));
		uiRect.SetVec4("rectangle", glm::vec4(0, 0, Window::Width, Window::Height));
		//uiRect.DrawArray(TriangleStrip, 4);
		//uiRect.DrawArray(TriangleStrip, 4);
	}

	DisableGLFeatures<StencilTesting>();
	//EnableGLFeatures<DepthTesting>();
	{
		DisablePushFlags(FaceCulling);
		glDepthMask(GL_FALSE);
		glDepthFunc(GL_LEQUAL);
		Shader& skyBox = ShaderBank::Retrieve("skyBox");
		skyBox.SetActiveShader();
		plainVAO.Bind();
		plainVAO.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"), 0);
		skyBox.SetTextureUnit("skyBox", sky);
		skyBox.DrawElements<DrawType::Triangle>(solidCubeIndex);
		glDepthFunc(GL_GEQUAL);
	}
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

	ShaderBank::Get("basic").SetActiveShader();
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	plainVAO.Bind();
	plainVAO.BindArrayBuffer(rayBuffer);
	ShaderBank::Get("basic").SetMat4("Model", glm::mat4(1.f));
	ShaderBank::Get("basic").SetVec4("Color", glm::vec4(1.f));
	ShaderBank::Get("basic").DrawArray(rayBuffer);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (false)
	{
		Shader& uiRectTexture = ShaderBank::Retrieve("uiRectTexture");
		uiRectTexture.SetActiveShader();

		auto& colored = buffet.GetColor();
		uiRectTexture.SetTextureUnit("image", colored, 0);
		glm::vec4 loc = glm::vec4((Window::Width - colored.GetWidth()) / 2, (Window::Height - colored.GetHeight()) / 2,
			colored.GetWidth(), colored.GetHeight());
		uiRectTexture.SetVec4("rectangle", loc);
		uiRectTexture.DrawArray<DrawType::TriangleStrip>(4);
		
		uiRectTexture.SetTextureUnit("image", (buttonToggle) ? buttonA : buttonB, 0);
		uiRectTexture.SetVec4("rectangle", buttonRect);
		uiRectTexture.DrawArray<DrawType::TriangleStrip>(4);

		uiRectTexture.SetTextureUnit("image", help.GetTexture(), 0);
		uiRectTexture.SetVec4("rectangle", help.GetRect());
		uiRectTexture.DrawArray<DrawType::TriangleStrip>(4);

		uiRectTexture.SetTextureUnit("image", normalMap);
		uiRectTexture.SetVec4("rectangle", { 0, 0, normalMap.GetSize()});
		//uiRectTexture.DrawArray<DrawType::TriangleStrip>(4);
		DisableGLFeatures<FaceCulling>();
		DisableGLFeatures<Blending>();
		
	}
	EnableGLFeatures<Blending>();
	// Debug Info Display
	{
		DisablePushFlags(DepthTesting);
		Shader& fontShader = ShaderBank::Retrieve("fontShader");
		fontShader.SetActiveShader();
		fontVAO.Bind();
		fontVAO.BindArrayBuffer(textBuffer);
		fontShader.SetTextureUnit("fontTexture", fonter.GetTexture(), 0);
		fontShader.DrawArray<DrawType::Triangle>(textBuffer);
	}

	DisableGLFeatures<Blending>();
	EnableGLFeatures<DepthTesting>();

	// Framebuffer stuff
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, renderTarget.GetFrameBuffer());
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glm::ivec2 dimension2 = Window::GetSize();
	// TODO: Let render resolution and screen resolution be decoupled
	glBlitFramebuffer(0, 0, dimension2.x, dimension2.y, 0, 0, dimension2.x, dimension2.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	BindDefaultFrameBuffer();

	glLineWidth(1.f);
	DisableGLFeatures<DepthTesting>();
	ShaderBank::Retrieve("widget").SetActiveShader();
	ShaderBank::Retrieve("widget").DrawArray<DrawType::Lines>(6);

	EnableGLFeatures<DepthTesting | StencilTesting | FaceCulling>();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	
	auto end = std::chrono::high_resolution_clock::now();
	displayTime = end - displayStartTime;
	displayStartTime = end;

	glEndQuery(GL_TIME_ELAPSED);
	glQueries.push_back(currentRenderQuery);
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

	static TimerAverage<300> displayTimes, idleTimes;
	static TimerAverage<300, float> frames;

	// This is in NANOseconds
	static TimerAverage<100, GLuint64> renderDelays;
	static CircularBuffer<float, 200> fpsPlot;
	static unsigned long long displaySimple = 0, idleSimple = 0;

	idleFrameCounter++;
	const TimePoint idleStart = std::chrono::high_resolution_clock::now();
	const TimeDelta delta = idleStart - lastIdleStart;

	const float timeDelta = std::chrono::duration<float, std::chrono::seconds::period>(delta).count();

	float averageFps = frames.Update(1.f / timeDelta);
	long long averageIdle = idleTimes.Update(idleTime.count() / 1000);
	long long averageDisplay = displayTimes.Update(displayTime.count() / 1000);

	fpsPlot.Push(timeDelta * 1000.f);
	static bool disableFpsDisplay = true;

	if (disableFpsDisplay)
	{
		ImGui::Begin("Metrics", &disableFpsDisplay);
		auto lienarFrames = fpsPlot.GetLinear();
		ImGui::PlotLines("##2", lienarFrames.data(), static_cast<int>(lienarFrames.size()), 0, "Frame Time", 0.f, 10.f, ImVec2(100, 100));
		ImGui::SameLine(); ImGui::Text(std::format("(ms): {:2.3}", 1000.f / averageFps).c_str());
		ImGui::End();
	}
	std::erase_if(glQueries,
		[&](GLuint query)
		{
			GLuint64 out = 0;
			glGetQueryObjectui64v(query, GL_QUERY_RESULT_NO_WAIT, &out);
			if (out)
			{
				renderDelays.Update(out);
				glDeleteQueries(1, &query);
				return true;
			}
			return false;
		}
	);

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
			windowShouldClose = true;
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
	if (debugFlags[DYNAMIC_TREE])
	{
		dynamicTreeBoxes = Level::GetBulletTree().GetBoxes();
		//dynamicTreeBoxes = Level::GetTriangleTree().GetBoxes();
	}

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
	buffered << '\n' << glm::dot(glm::normalize(playfield.GetVelocity()), playfield.GetModel().rotation * World::Forward);
	Level::SetInterest(management.GetPos());
	
	constexpr auto formatString = "FPS:{:7.2f}\nTime:{:4.2f}ms\nIdle:{}ns\nDisplay: {}us\n-Concurrent: {}us\
		\n-GPU Block Time: {}us\nAverage Tick Length:{}us\nMax Tick Length:{:4.2f}ms\nTicks/Second: {:7.2f}\n{}";

	auto renderTime = renderDelays.Get() / 1000;
	auto currentRenderDelay = renderDelays.Get() / 1000;
	if (static_cast<long long>(renderTime) < averageDisplay)
	{
		currentRenderDelay = 0;
	}
	else
	{
		currentRenderDelay -= averageDisplay;
	}

	std::string formatted = std::format(formatString, averageFps, 1000.f / averageFps, averageIdle, renderTime,
		averageDisplay, currentRenderDelay, averageTickTime, maxTickTime / 1000.f, gameTicks / glfwGetTime(), buffered.str());

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

		playfield.Update(boardState);
		const Model playerModel = playfield.GetModel();
		const Frustum localFrust = GetFrustum(playerModel);

		// Bullet stuff;
		std::vector<glm::mat4> inactive, blarg;

		std::vector<LightVolume> volumes{ constantLights };
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
			volumes.push_back({ glm::vec4(point, 20.f), glm::vec4(120.f,204.f,226.f, 1.f) / 255.f, glm::vec4(1.f, 0.5f, 0.05f, 1.f) });
		}
		shieldPos.Swap(tmep);

		// TODO: I combined these but it's sloooooow
		std::lock_guard opinion(bulletMutex);
		[[maybe_unused]] std::size_t removedBullets = Level::GetBulletTree().FullService([&](Bullet& local)
			{
				if (!local.IsValid())
				{
					return REMOVE;
				}
				glm::vec3 previous = local.transform.position;
				if (!debugFlags[FREEZE_GAMEPLAY])
				{
					local.Update();
				}
				const OBB transformedBox = local.GetOBB();
				const AABB endState = transformedBox.GetAABB();

				if (!localFrust.Overlaps(Sphere(endState.GetCenter(), glm::compMax(endState.Deviation()))))
				{
					inactive.push_back(local.GetModel().GetModelMatrix());
					blarg.push_back(transformedBox.GetModelMatrix());
					if (local.lifeTime > 10)
					{
						volumes.push_back({ glm::vec4(local.transform.position, 15.f), glm::vec4(1.f, 1.f, 0.f, 1.f), glm::vec4(1.f, 0.f, 0.05f, 1.f) });
					}
				}
				if (previous == local.transform.position)
				{
					return DO_NOTHING;
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
								return REMOVE;
							}
						}
					}
				}

				for (const auto& currentTri : Level::GetTriangleTree().Search(endState))
				{
					if (DetectCollision::Overlap(transformedBox, *currentTri))
					{
						// Don't let enemy decals clog things up 
						if (local.team != 0)
							return REMOVE;
						// TODO: change this so that the output vector isn't the big list so the actual generation of the decals
						// can be parallelized, with only the copying needing sequential access
						// If no decals were generated, then it didn't 'precisely' overlap any of the geometry, and as
						// generating decals also requires a OctTreeSearch, escape the outer one.
						if (false && decalVertex.ExclusiveOperation(
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
							return REMOVE;
						}
						break;
					}
				}
				if (previous != local.transform.position)
				{
					return RESEAT;
				}
				return DO_NOTHING;
			}
		);
		// Maybe this is a "better" method of syncing stuff than the weird hack of whatever I had before
		bulletMatricies.Swap(inactive);
		if (blarg.size() > 0)
		{
			bulletImpacts.Swap(blarg);
		}
		std::vector<LightVolume> volumer{ };
		std::ranges::copy(volumes | std::ranges::views::all, std::back_inserter(volumer));
		drawingVolumes.Swap(volumes);
		
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
		//float playerSpeed = glm::length(playfield.GetVelocity());
		//const glm::vec3 playerForward = playfield.GetVelocity();

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
	} while (!windowShouldClose);
}

void window_focus_callback([[maybe_unused]] GLFWwindow* window, int focused)
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
				copied = static_cast<unsigned char>(std::tolower(copied));
			else if (!(mods & GLFW_MOD_SHIFT)) 
				copied = static_cast<unsigned char>(std::tolower(copied));
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
			//if (magnetic.Finished())
			{
				//magnetic.Start({ playfield.GetModel().translation, playfield.GetModel().rotation});
			}
			std::lock_guard lockon(bulletMutex);
			std::mt19937 engineer;
			std::uniform_real_distribution<float> numbers(-150.f, 150.f);
			std::uniform_real_distribution<float> smallNumbers(-1.f, 1.f);
			for (int i = 0; i < 150; i++)
			{
				glm::vec3 position(numbers(engineer), numbers(engineer), numbers(engineer));
				glm::vec3 direction = glm::normalize(glm::vec3(smallNumbers(engineer), smallNumbers(engineer), smallNumbers(engineer)));
				Level::AddBulletTree(position, direction * 100.f, World::Up, 0);
			}
		}
		if (key == GLFW_KEY_U)
		{
			addExplosion++;
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
			windowShouldClose = true;
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

	// Get the camera orientation
	glm::vec3 radians = glm::radians(glm::vec3(0.f));

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

void mouseButtonFunc(GLFWwindow* window, int button, int action, [[maybe_unused]] int status)
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

void mouseScrollFunc([[maybe_unused]] GLFWwindow* window, [[maybe_unused]] double xDelta, double yDelta)
{
	playerSpeedControl += 0.1f * glm::sign(static_cast<float>(yDelta));
	playerSpeedControl = glm::clamp(playerSpeedControl, 0.f, 1.f);
}

void mouseCursorFunc([[maybe_unused]] GLFWwindow* window, double xPos, double yPos)
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
		// TODO: Sensitivity values

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

void window_size_callback([[maybe_unused]] GLFWwindow* window, int width, int height)
{
	Window::Update(width, height);
	
	cameraUniformBuffer.Generate(DynamicDraw, 2 * sizeof(glm::mat4));
	cameraUniformBuffer.SetBindingPoint(0);
	cameraUniformBuffer.BindUniform();

	const glm::mat4 projection = Window::GetPerspective(zNear, zFar);
	cameraUniformBuffer.BufferSubData(projection, sizeof(glm::mat4));

	FilterStruct screenFilters{ MinLinear, MagLinear, BorderClamp, BorderClamp };

	screenSpaceBuffer.Generate(StaticRead, sizeof(glm::mat4));
	screenSpaceBuffer.SetBindingPoint(1);
	screenSpaceBuffer.BindUniform();
	screenSpaceBuffer.BufferSubData(Window::GetOrthogonal());

	earlyDepth.GetColor().CreateEmpty(glm::ivec2(1), InternalRed8);
	earlyDepth.GetDepth().CreateEmpty(Window::GetSizeF() / EarlyDepthRatio, InternalDepthFloat32);
	earlyDepth.GetDepth().SetFilters(MinNearest, MagNearest, EdgeClamp, EdgeClamp);
	earlyDepth.Assemble();

	renderTarget.GetColor().CreateEmpty(Window::GetSize(), InternalRGBA8);
	renderTarget.GetDepth().CreateEmpty(Window::GetSize(), InternalDepthFloat32);
	renderTarget.Assemble();

	// This is dependent on screen size so must be here.
	{
		// TODO: Put the constants and stuff in here so it doesn't have to be recompiled all the time
		QUICKTIMER("Foolhardy");
		ShaderBank::Get("lightCulling").CompileCompute("light_cull");
		Shader& shader = ShaderBank::Get("computation");
		shader.CompileCompute("compute_frustums");
		shader.UniformBlockBinding("Camera", 0);

		auto nextMult = [](auto a, auto b) {return glm::ceil(a / b) * b; };

		// Moving past the sample
		shader.SetActiveShader();

		// ???? <- Oh it's about reserving sizes of things, grumble grumble
		struct A{glm::vec3 c;float b;};
		struct B{A ar[4];};
		// Frustum space calculations
		auto amount = nextMult(Window::GetSizeF(), gridResolution) / gridResolution;
		numTiles = static_cast<decltype(numTiles)>(amount.x * amount.y);
		tileDimension = amount;
		//ShaderStorage::Get("Frustums");      // 5
		//ShaderStorage::Get("LightIndicies"); // 6
		//ShaderStorage::Get("LightGrid");     // 7
		//ShaderStorage::Get("LightBlock");    // 8
		//ShaderStorage::Get("LightGrid2");    // 9
		// Frustums
		ShaderStorage::Get("Frustums").BufferData(StaticVector<B>(numTiles));
		ShaderStorage::Get("Frustums").BindBufferBase(5);
		// Say 100 light max per tile
		ShaderStorage::Get("LightIndicies").Reserve(sizeof(std::uint32_t) * 100 * numTiles);
		ShaderStorage::Get("LightIndicies").BindBufferBase(6);
		// Only need one per tile
		ShaderStorage::Get("LightGrid").BufferData(StaticVector<glm::uvec2>(numTiles));
		ShaderStorage::Get("LightGrid").BindBufferBase(7);
		// Light Block is dynamically generated, but dummy data will suffice
		ShaderStorage::Get("LightBlock").BufferData<std::uint32_t>(0);
		ShaderStorage::Get("LightBlock").BindBufferBase(8);
		// Must be reset every frame before usage
		ShaderStorage::Get("LightGrid2").BufferData(std::to_array({0u, 0u}));
		ShaderStorage::Get("LightGrid2").BindBufferBase(9);
		ShaderStorage::Get("LightBlockOriginal").BufferData(std::to_array({ 0u, 0u }));
		ShaderStorage::Retrieve("LightBlockOriginal").BindBufferBase(10);
		{
			UniformBuffer& uniformed = Bank<UniformBuffer>::Get("ForwardPlusConstants");
			//uniformed.Generate(StaticDraw, sizeof(glm::mat2) + sizeof(glm::uvec2) + sizeof(glm::vec2) + sizeof(int));
			uniformed.Generate(StaticDraw, 64);
			glm::mat2 smp = GetLower2x2(glm::inverse(projection));
			uniformed.BufferSubData(smp[0], 0);
			uniformed.BufferSubData(smp[0], sizeof(glm::vec2));
			uniformed.BufferSubData(smp[1], sizeof(glm::vec4));
			uniformed.BufferSubData(smp[1], sizeof(glm::vec4) + sizeof(glm::vec2));
			uniformed.BufferSubData(Window::GetSizeF(), sizeof(glm::mat4x2));
			uniformed.BufferSubData(tileDimension, sizeof(glm::mat4x2) + sizeof(glm::vec2));
			uniformed.BufferSubData(static_cast<int>(gridResolution), sizeof(glm::mat4x2) + sizeof(glm::vec2) + sizeof(glm::uvec2));
			uniformed.BufferSubData(glm::uvec3(666, 1337, 547), 52);
			uniformed.SetBindingPoint(2);
			uniformed.BindUniform();
		}
		shader.UniformBlockBinding("ForwardPlusConstants", 2);
		shader.SetMat4("InverseProjection", glm::inverse(projection));
		shader.DispatchCompute(tileDimension.x, tileDimension.y);
		ShaderBank::Get("lightCulling").UniformBlockBinding("ForwardPlusConstants", 2);
	}
}

void init();

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
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
	TestFunc();
	//Input::ControllerStuff();
	//testOBB();
	std::srand(NULL);
	// OpenGL Feature Enabling
	EnableGLFeatures<DepthTesting | FaceCulling | DebugOutput>();
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	DisableGLFeatures<MultiSampling>();

	//glDepthFunc(GL_LEQUAL);
	glDepthFunc(GL_GEQUAL);

	glClearColor(0, 0, 0, 1);
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
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
	Shader::IncludeInShaderFilesystem("forward_buffers", "forward_buffers.incl");
	Shader::IncludeInShaderFilesystem("forward_plus", "forward_plus.incl");
	Shader::IncludeInShaderFilesystem("cone", "cone.incl");
	ExternalShaders::Setup();

	ShaderBank::Get("basic").CompileSimple("basic");
	ShaderBank::Get("bulletShader").Compile("color_final", "mesh_final");
	ShaderBank::Get("debris").Compile("mesh_final_instance", "mesh_final");
	ShaderBank::Get("skyBox").CompileSimple("sky");
	ShaderBank::Get("decalShader").CompileSimple("decal");
	
	ShaderBank::Get("engine").CompileSimple("engine");
	ShaderBank::Get("fontShader").CompileSimple("font");
	ShaderBank::Get("nineSlicer").CompileSimple("ui_nine");
	ShaderBank::Get("pathNodeView").CompileSimple("path_node");
	ShaderBank::Get("stencilTest").CompileSimple("stencil_");
	ShaderBank::Get("trail").CompileSimple("trail");
	ShaderBank::Get("uiRect").CompileSimple("ui_rect");
	ShaderBank::Get("uiRectTexture").CompileSimple("ui_rect_texture");
	ShaderBank::Get("uniform").CompileSimple("uniform");
	ShaderBank::Get("uniform").CompileSimple("uniform");
	ShaderBank::Get("vision").CompileSimple("vision");
	ShaderBank::Get("widget").CompileSimple("widget");

	ShaderBank::Get("ShieldTexture").Compile(
		"framebuffer", "shield_texture"
	);
	
	Shader::DefineTemp("#define DEBRIS_COUNT 666");
	ShaderBank::Get("debrisCompute").CompileCompute("debris_compute");

	ShaderBank::Get("dither").CompileSimple("light_text_dither");
	ShaderBank::Get("depthOnly").Compile("new_mesh_simp", "empty");

	Shader::ForceRecompile(true);
	ShaderBank::Get("forwardPlus").Compile("new_mesh_single", "forward_plus");
	ShaderBank::Get("forwardPlusMulti").Compile("new_mesh", "forward_plus");
	ShaderBank::Get("dust").CompileSimple("dust");
	Shader::ForceRecompile(false);

	ShaderBank::Get("uniformInstance").Compile("uniform_instance", "uniform");
	ShaderBank::Get("combinePass").Compile("framebuffer", "combine_pass");
	ShaderBank::Get("visualize").Compile("framebuffer", "visualize");
	ShaderBank::Get("Shielding").CompileSimple("shield");
	ShaderBank::Get("ship").CompileSimple("mesh_final");

	ShaderBank::for_each(std::to_array({ "depthOnly", "dust", "forwardPlus", "forwardPlusMulti", "engine",
		"uniformInstance", "Shielding", "debris", "bulletShader", "skyBox", "ship", "decalShader", "basic", "vision",
		"trail", "uniform", "pathNodeView", "stencilTest", "debrisCompute"}),
		[](auto& element)
		{
			element.UniformBlockBinding("Camera", 0);
		}
	);

	ShaderBank::for_each(std::to_array({ "dust", "forwardPlus", "forwardPlusMulti", "visualize"}),
		[](auto& element)
		{
			element.UniformBlockBinding("ForwardPlusConstants", 2);
		}
	);

	ShaderBank::for_each(std::to_array({ "fontShader", "uiRect", "uiRectTexture", "nineSlicer"}),
		[](auto& element)
		{
			element.UniformBlockBinding("ScreenSpace", 1);
		}
	);

	// VAO SETUP
	fontVAO.ArrayFormat<UIVertex>();

	meshVAO.ArrayFormat<MeshVertex>();
	nineSliced.ArrayFormatOverride<glm::vec4>("rectangle", ShaderBank::Retrieve("nineSlicer"), 0, 1);

	pathNodeVAO.ArrayFormat<Vertex>(0);
	pathNodeVAO.ArrayFormatOverride<glm::vec3>("Position", ShaderBank::Retrieve("pathNodeView"), 1, 1);
	plainVAO.ArrayFormat<Vertex>();
	VAOBank::Get("uniform").ArrayFormat<Vertex>();
	VAOBank::Get("meshVertex").ArrayFormat<MeshVertex>();
	VAOBank::Get("normalVertex").ArrayFormat<NormalVertex>();

	VAOBank::Get("engineInstance").ArrayFormatOverride<glm::vec4>(0, 0, 1);
	VAOBank::Get("muscle").ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, 56);
	//VAOBank::Get("Debris").ArrayFormatOverride<glm::vec4>(0, 0, 1);
	{
		VAO& ref = VAOBank::Get("bigscrem");
		ref.ArrayFormatOverride<glm::vec4>(0, 0, 1);
	}
	{
		VAO& ref = VAOBank::Get("uniformInstance");
		ref.ArrayFormat<Vertex>();
		ref.ArrayFormatM<glm::mat4>(ShaderBank::Get("uniformInstance"), 1, 1, "Model");
	}
	{
		VAO& ref = VAOBank::Get("forwardPlusMulti");
		ref.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(NormalMeshVertex, normal), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(2, 0, 0, offsetof(NormalMeshVertex, tangent), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(3, 0, 0, offsetof(NormalMeshVertex, biTangent), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::vec2>(4, 0, 0, offsetof(NormalMeshVertex, texture), sizeof(NormalMeshVertex));
		ref.ArrayFormatOverride<glm::mat4>("modelMat", ShaderBank::Get("forwardPlusMulti"), 1, 1, 0, sizeof(MeshMatrix));
		ref.ArrayFormatOverride<glm::mat4>("normalMat", ShaderBank::Get("forwardPlusMulti"), 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
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

	texturedVAO.ArrayFormat<TextureVertex>();

	colorVAO.ArrayFormat<ColoredVertex>();

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


	// RAY SETUP
	std::array<glm::vec3, 20> rays = {};
	rays.fill(glm::vec3(0));
	rayBuffer.BufferData(rays);

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

	// Cube map shenanigans
	{
		// From Here https://opengameart.org/content/space-skybox-1 under CC0 Public Domain License
		sky.Generate(std::to_array<std::string>({"skybox/space_ft.png", "skybox/space_bk.png", "skybox/space_up.png", 
			"skybox/space_dn.png", "skybox/space_rt.png", "skybox/space_lf.png"}));
	}

	tickTockMan.Init();
	Parallel::SetStatus(true);

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
		//Bullet::Collision.ReCenter(Bullet::Collision.Forward());
		//Bullet::Collision.ReCenter(Bullet::Collision.Forward() * Bullet::Collision.GetScale().x);
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

	{
		std::array<glm::vec4, 20 * 2> lightingArray{ glm::vec4(0.f) };
		for (std::size_t i = 0; i < lightingArray.size(); i += 2)
		{
			Triangle parent = nodeTri[rand() % nodeTri.size()];
			lightingArray[i] = glm::vec4(parent.GetCenter() + parent.GetNormal() * glm::max(glm::gaussRand(25.f, 5.f), 5.f), 
				glm::min(glm::gaussRand(25.f, 10.f), 40.f));
			lightingArray[i + 1] = glm::vec4(glm::abs(glm::ballRand(1.f)), 0.f);
			
			//std::cout << lightingArray[i] << ":" << lightingArray[i + 1] << '\n';
			constantLights.push_back({ lightingArray[i], lightingArray[i + 1], glm::vec4(1.f, 1.f / 30.f, 0.002f, 1.f) });
			if (!bp.TestPoint(glm::vec3(lightingArray[i])))
			{
				std::cout << "Invalid Point" << '\n';
			}
		}
		globalLighting.BufferData(lightingArray);
		globalLighting.SetBindingPoint(4);
		globalLighting.BindUniform();
		Bank<ArrayBuffer>::Get("dummy").BufferData(std::array<glm::vec3, 4>());
		Bank<ArrayBuffer>::Get("dummy2").BufferData(std::array<glm::vec3, 10>());
	}
	
	{
		QUICKTIMER("Debris Initializing");
		struct doub { glm::vec4 pos, dir; };
		StaticVector<doub> hmm2(dustCount);
		for (std::size_t i = 0; i < hmm2.size(); i++)
		{
			hmm2[i].pos = glm::vec4(glm::ballRand(zFar / 2.f) + playfield.GetModel().translation * 10000.f, 1.f);
			// Might need some work
			hmm2[i].dir = glm::vec4(Tick::TimeDelta * glm::gaussRand(0.4f, 0.225f) * glm::sphericalRand(1.f), 1.f);
		}
		ShaderStorage::Get("RawDebris").BufferData(hmm2);
		ShaderStorage::Get("DrawDebris").Reserve(sizeof(glm::vec4) * hmm2.size());
		ShaderStorage::Get("DebrisIndirect").Reserve(sizeof(DrawIndirect));
		BufferBank::Get("DrawDebris").Reserve(sizeof(glm::vec4) * hmm2.size());
		Bank<DrawIndirectBuffer>::Get("DebrisIndirect").Reserve(sizeof(DrawIndirect));
	}

	for (int i = 0; i < 10; i++)
	{
		auto& foo = management.Make();
		foo.Init(i > 5 ? glm::vec3(0.f, 60.f, 0.f) : glm::vec3(0.f, -60.f, 0.f));
	}
	//Level::GetTriangleTree().UpdateStructure();

	{
		QUICKTIMER("AABB Stress test");
		std::size_t succeed = 0, fails = 0;
		Level::GetTriangleTree().for_each(
			[&](auto& ref2) 
			{
				auto& ref = *ref2;
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

	// =============================================================

	{
		QuickTimer _tim{ "Sphere/Capsule Generation" };
		Sphere::GenerateMesh(sphereBuffer, sphereIndicies, 100, 100);
		Capsule::GenerateMesh(capsuleBuffer, capsuleIndex, 0.75f, 3.25f, 30, 30);
	}

	bulletVAO.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0);
	bulletVAO.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(ColoredVertex, color));
	bulletVAO.ArrayFormatOverride<glm::mat4>("modelMat", ShaderBank::Get("bulletShader"), 1, 1, 0, sizeof(glm::mat4));

	// TODO: Figure out why std::move(readobj) has the wrong number of elements
	//std::cout << satelitePairs.size() << ":\n";
	Font::SetFontDirectory("Fonts");
	
	DebrisManager::LoadResources();
	trashMan.Init();
	Satelite::LoadResources();

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
		Shader& nineSlicer = ShaderBank::Retrieve("nineSlicer");
		Shader& uiRectTexture = ShaderBank::Retrieve("uiRectTexture");
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