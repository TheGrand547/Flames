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
#include <functional>
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
#include "entities/Lights.h"
#include "entities/DecayLight.h"
#include "entities/Laser.h"

// TODO: https://github.com/zeux/meshoptimizer once you use meshes
// TODO: EASTL

ASCIIFont fonter;

// Buffers
ArrayBuffer textBuffer, sphereBuffer, stickBuffer;
ArrayBuffer exhaustBuffer;

MeshData guyMeshData;

ElementArray cubeOutlineIndex, solidCubeIndex, sphereIndicies, stickIndicies;

UniformBuffer cameraUniformBuffer, screenSpaceBuffer;

// Textures
Texture2D ditherTexture, normalMap, texture;
Texture2D buttonA, buttonB, nineSlice;
CubeMap sky;

// Not explicitly tied to OpenGL Globals

static unsigned int idleFrameCounter = 0;

constexpr auto FREEZE_GAMEPLAY = 1;
constexpr auto TIGHT_BOXES = 2;
constexpr auto CHECK_LIGHT_TILES = 3;
constexpr auto CHECK_LIGHT_VOLUMES = 4;
constexpr auto DYNAMIC_TREE = 5;
constexpr auto PRIMITIVE_COUNTING = 6;
// One for each number key
std::array<bool, '9' - '0' + 1> debugFlags{};

// Input Shenanigans
constexpr auto ArrowKeyUp = 0;
constexpr auto ArrowKeyDown = 1;
constexpr auto ArrowKeyRight = 2;
constexpr auto ArrowKeyLeft = 3;

std::array<bool, UCHAR_MAX> keyState{}, keyStateBackup{};

ColorFrameBuffer playerTextEntry;
std::stringstream letters("abc");
bool reRenderText = true;

constexpr float ANGLE_DELTA = 4;

float zNear = 0.1f, zFar = 1000.f;

static bool featureToggle = false;
static std::chrono::nanoseconds idleTime, displayTime;

// TODO: Semaphore version of buffersync
BufferSync<std::vector<TextureVertex>> decalVertex;

ArrayBuffer ui_tester_buffer;

std::vector<AABB> dynamicTreeBoxes;
using namespace Input;

static bool windowShouldClose = false;
static bool gameTickDone = false;

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

Satelite groovy{ glm::vec3(10.f, 10.f, 0) };
bool shiftHeld;
std::atomic_uchar addExplosion;

DebrisManager trashMan;

MeshData playerMesh;
MeshData bulletMesh;

BufferSync<std::vector<glm::mat4>> bulletMatricies, bulletBoxes;

MeshData levelGeometry;

static GLFWwindow* windowPointer = nullptr;

const float BulletDecalScale = 4.f;

BufferSync<std::vector<LightVolume>> drawingVolumes;
std::vector<LightVolume> constantLights;

glm::vec3 GetCameraFocus(const Model& playerModel, const glm::vec3& velocity)
{
	return playerModel.translation + (playerModel.rotation * glm::vec3(1.f, 0.f, 0.f)) * (10.f + Rectify(glm::length(velocity)) / 2.f);
}

std::pair<glm::vec3, glm::vec3> CalculateCameraPositionDir(const Model& playerModel)
{
	glm::vec3 localCamera{};
	const glm::vec3 velocity = playfield.GetVelocity();
	glm::vec3 basePoint = glm::vec3(-4.f, 2.5f, 0.f);
	
	localCamera = playerModel.rotation * basePoint;
	localCamera += playerModel.translation;

	const glm::vec3 modelForward = playerModel.rotation * glm::vec3(1.f, 0.f, 0.f);

	float speed = glm::length(velocity);

	if (speed > EPSILON)
	{
		// TODO: Max Speed
		float adjustedOffset = glm::mix(0.f, 1.5f, speed / 60.f);
		adjustedOffset = std::clamp(speed / 20.f, 0.f, 1.5f);
		localCamera -= glm::normalize(velocity) * adjustedOffset;
		//localCamera -= velocity / 20.f;
	}

	//const glm::vec3 cameraFocus = GetCameraFocus(playerModel, velocity);
	const glm::vec3 cameraFocus = localCamera + modelForward * (10.f + Rectify(glm::length(velocity)));
	const glm::vec3 cameraForward = glm::normalize(cameraFocus - localCamera);
	return { localCamera, cameraForward };
}

Frustum GetFrustum(const Model& playerModel)
{
	auto cameraPair = CalculateCameraPositionDir(playerModel);
	return Frustum(cameraPair.first, ForwardDir(cameraPair.second, playerModel.rotation * glm::vec3(0.f, 1.f, 0.f)), glm::vec2(zNear, zFar));
}

struct quickLight
{
	glm::vec3 start, end;
	std::uint32_t lifeTime;
};
BufferSync<std::vector<glm::vec3>> quickLightPairs;
std::vector<quickLight> zoopers;

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

static std::vector<GLuint> glFrameTimeQueries;
static GLuint particleCountQuery = 0;
static GLuint VertexCountQuery = 0;
std::mutex bulletMutex;

constexpr std::size_t dustDimension = 20;
constexpr std::size_t dustCount = dustDimension * dustDimension * dustDimension;

const glm::vec3 flashLightColor = glm::vec3(148.f, 252.f, 255.f) / 255.f;

std::vector<DecayLight> decaymen;

constexpr std::uint32_t MaxLights = 100;
// Get the number of buckets per tile, Adding an additional bucket for misc data
constexpr std::uint32_t BucketsPerTile = MaxLights / 32 + (MaxLights % 32 != 0) + 1;

static const glm::vec3 ShieldColor = glm::vec3(120.f, 204.f, 226.f) / 255.f;

void BindDrawFramebuffer()
{
	renderTarget.BindDraw();
	Window::Viewport();
}
struct infinite_pain
{
	glm::vec4 position, velocity, normal, color;
};
constexpr std::size_t Max_Partilces = 1024;
static BufferSync<std::vector<infinite_pain>> particlesNew;
static BufferSync<std::vector<glm::mat4>> bigCringe;

static bool timeCurrentQuery = false;
static std::atomic_flag tickHappend = ATOMIC_FLAG_INIT;


void display()
{
	bool vertexQueryEnable = debugFlags[PRIMITIVE_COUNTING];

	GLuint currentRenderQuery = 0;
	glGenQueries(1, &currentRenderQuery);
	glBeginQuery(GL_TIME_ELAPSED, currentRenderQuery);
	if (vertexQueryEnable)
	{
		glBeginQuery(GL_PRIMITIVES_GENERATED, VertexCountQuery);
	}

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
	DefaultDepthTest();
	ClearFramebuffer<ColorBuffer | DepthBuffer>();

	const Model playerModel(playfield.GetModel());

	// Camera matrix
	const glm::mat3 axes(playerModel.rotation);
	const glm::vec3 velocity = playfield.GetVelocity();
	
	std::pair<glm::vec3, glm::vec3> cameraPair = CalculateCameraPositionDir(playerModel);
	const glm::vec3 localCamera = cameraPair.first;
	const glm::vec3 cameraForward = cameraPair.second;

	const glm::mat4 view = glm::lookAt(localCamera, GetCameraFocus(playerModel, velocity), axes[1]);
	cameraUniformBuffer.BufferSubData(view, 0);

	Shader& uniform = ShaderBank::Retrieve("uniform");
	drawingVolumes.ExclusiveOperation(
		[&](std::vector<LightVolume>& data)
		{
			// The players 'torch'
			LightVolume greeblies;
			constexpr float FlashLightHeight = 200.f;
			// The full range from edge to edge
			constexpr float FlashLightAngle = 50.f;

			greeblies.position = glm::vec4(playerModel.translation, 1.f);
			greeblies.color = glm::vec4(flashLightColor, 1.f);
			greeblies.constants = glm::vec4(1.f, 1.f / 200.f, 1.f / 2000.f, 1.f);
			greeblies.direction = glm::vec4(axes[0], 1.f);
			ConeLightingInfo(greeblies, FlashLightHeight, FlashLightAngle);
			
			std::vector<BigLightVolume> biggest;
			biggest.reserve(data.size() + 1);
			biggest.push_back(MakeBig(greeblies, view));
			std::ranges::copy(data | std::views::transform(
				[&](const LightVolume& v)
				{
					return MakeBig(v, view);
				}
			),
				std::back_inserter(biggest));
			ShaderStorage::Retrieve("LightBlock").BufferData(biggest);
			std::vector<glm::mat4> sloppyCode;
			std::ranges::copy(data | std::views::transform(
				[&](const LightVolume& v)
				{
					Model career(v.position);
					career.scale = glm::vec3(v.position.w);
					return career.GetModelMatrix();
				}
			),
				std::back_inserter(sloppyCode));
			BufferBank::Get("uniformInstanceSphere").BufferData(sloppyCode);
		}
	);
	// Compute Shaders
	{
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		Shader& cullLights = ShaderBank::Retrieve("lightCulling");
		cullLights.SetActiveShader();
		cullLights.SetTextureUnit("DepthBuffer", earlyDepth.GetDepth(), 0);
		cullLights.SetUnsignedInt("featureToggle", featureToggle);
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
		
		glUseProgram(0);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		Shader& lowester = ShaderBank::Retrieve("particleCompute");

		lowester.SetActiveShader();
		auto& rawParts = ShaderStorage::Retrieve("RawParticles");
		auto& drawParts = ShaderStorage::Retrieve("DrawParticles");
		auto& indirectParts = ShaderStorage::Retrieve("IndirectParticles");
		rawParts.BindBufferBase(0);
		drawParts.BindBufferBase(1);
		indirectParts.BindBufferBase(2);
		particlesNew.ExclusiveOperation([](auto& in)
			{
				ShaderStorage::Retrieve("MiscParticles").BufferSubData(static_cast<unsigned int>(in.size()), sizeof(unsigned int));
				if (in.size() > 0)
				{
					ShaderStorage::Retrieve("NewParticles").BufferData(in);
					in.clear();
				}
			});
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		ShaderStorage::Retrieve("NewParticles").BindBufferBase(3);
		ShaderStorage::Retrieve("MiscParticles").BindBufferBase(4);
		lowester.SetUnsignedInt("pauseMotion", debugFlags[FREEZE_GAMEPLAY]);
		lowester.DispatchCompute(Max_Partilces / 64);

		auto& drawOutParts = BufferBank::Retrieve("DrawParticles");
		auto& indirectOutParts = Bank<DrawIndirectBuffer>::Retrieve("IndirectParticles");

		// END OF COMPUTE SECTION
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);



		auto& outputBuffer = BufferBank::Retrieve("DrawDebris");
		auto& outputIndirect = Bank<DrawIndirectBuffer>::Retrieve("DebrisIndirect");
		glCopyNamedBufferSubData(transformedOut.GetBuffer(), outputBuffer.GetBuffer(), 0, 0, transformedOut.Size());
		glCopyNamedBufferSubData(indirectOut.GetBuffer(), outputIndirect.GetBuffer(), 0, 0, sizeof(unsigned int) * 4);

		glCopyNamedBufferSubData(drawParts.GetBuffer(), drawOutParts.GetBuffer(), 0, 0, drawParts.Size());
		glCopyNamedBufferSubData(indirectParts.GetBuffer(), indirectOutParts.GetBuffer(), 0, 0, sizeof(unsigned int) * 4);
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

	interzone.DrawElements<DrawType::Triangle>(levelGeometry.indirect);

	Level::GetShips().Draw(guyMeshData, outerzone, interzone);
	bobert.Draw(interzone, outerzone);
	
	auto& buf = BufferBank::Get("player");
	Model playerDrawModel = playerModel;
	playerDrawModel.scale *= 0.5f;
	buf.BufferData(playerDrawModel.GetMatrixPair());
	outerzone.BindArrayBuffer(buf, 1);
	playfield.Draw(interzone, outerzone, playerMesh, playerModel);
	// Dust particles
	{
		// *Needs* to have the copying between the buffers completed by this point
		glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
		Shader& local = ShaderBank::Retrieve("dust");
		local.SetActiveShader();
		VAO& vao = VAOBank::Retrieve("dust");
		vao.Bind();
		vao.BindArrayBuffer(BufferBank::Retrieve("DrawDebris"), 0);
		local.SetVec3("shapeColor", glm::vec3(0.9f));
		local.DrawArrayIndirect<DrawType::TriangleStrip>(Bank<DrawIndirectBuffer>::Retrieve("DebrisIndirect"));
	}

	if (debugFlags[CHECK_LIGHT_TILES])
	{
		DisablePushFlags(DepthTesting | FaceCulling | Blending);
		DisableDepthWritePush;
		Shader& sahder = ShaderBank::Get("visualize");
		sahder.SetActiveShader();
		static int thresholdAmount = 10;
		ImGui::Begin("Light Threshold");
		ImGui::SliderInt("Threshold", &thresholdAmount, 0, 100);
		ImGui::End();
		sahder.SetInt("maxLight", thresholdAmount);
		sahder.DrawArray<DrawType::TriangleStrip>(4);
	}

	// Copy the current depth buffer status to the early depth buffer for next frame
	glBindFramebuffer(GL_READ_FRAMEBUFFER, renderTarget.GetFrameBuffer());
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, earlyDepth.GetFrameBuffer());
	glm::ivec2 dimension = Window::GetSize();
	glm::ivec2 depthSize = earlyDepth.GetDepth().GetSize();
	glBlitFramebuffer(0, 0, dimension.x, dimension.y, 0, 0, depthSize.x, depthSize.y, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderTarget.GetFrameBuffer());

	// DO NOT TOUCH
	{
		/* STICK FIGURE GUY */
		uniform.SetActiveShader();
		VAOBank::Retrieve("uniform").DoubleBindArrayBuffer(stickBuffer);

		glm::vec3 colors = glm::vec3(1, 0, 0);
		Model m22(glm::vec3(10, 0, 0));
		uniform.SetMat4("Model", m22.GetModelMatrix());
		uniform.SetVec3("color", colors);
		uniform.DrawElements<DrawType::LineStrip>(stickIndicies);
	}

	if (debugFlags[DYNAMIC_TREE])
	{
		// This *could* be done more nicely with uniformInstance, But in this case it'd be more hassle than it's worth
		uniform.SetActiveShader();
		glm::vec3 blue(0, 0, 1);
		VAOBank::Retrieve("uniform").DoubleBindArrayBuffer(Bank<ArrayBuffer>::Retrieve("plainCube"));
		uniform.SetVec3("color", glm::vec3(1, 0.65, 0));
		for (auto& box : dynamicTreeBoxes)
		{
			auto d = box.GetModel();
			d.scale *= 0.99f;
			uniform.SetMat4("Model", d.GetModelMatrix());
			uniform.DrawElements<DrawType::Lines>(cubeOutlineIndex);
		}
	}
	if (debugFlags[CHECK_LIGHT_VOLUMES])
	{
		Shader& shader = ShaderBank::Get("uniformInstance");
		VAO& vao = VAOBank::Get("uniformInstanceSphere");
		shader.SetActiveShader();
		vao.Bind();
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		vao.BindArrayBuffer(sphereBuffer, 0);
		vao.BindArrayBuffer(BufferBank::Get("uniformInstanceSphere"), 1);
		shader.DrawElementsInstanced<DrawType::Triangle>(sphereIndicies, BufferBank::Get("uniformInstanceSphere"));
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	if (debugFlags[TIGHT_BOXES])
	{
		Shader& shader = ShaderBank::Get("uniformInstance");
		VAO& vao = VAOBank::Retrieve("uniformInstance");
		ArrayBuffer& instances = BufferBank::Get("tightBoxes");
		shader.SetActiveShader();
		vao.Bind();
		vao.BindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"), 0);
		vao.BindArrayBuffer(instances, 1);
		shader.DrawElementsInstanced<DrawType::Lines>(cubeOutlineIndex, instances);

		ArrayBuffer& bulletBoxes = Bank<ArrayBuffer>::Retrieve("bulletBoxes");
		vao.BindArrayBuffer(bulletBoxes, 1);
		shader.DrawElementsInstanced<DrawType::Lines>(cubeOutlineIndex, bulletBoxes);
	}

	{
		Shader& shader = ShaderBank::Get("laserBatch");
		VAO& vao = VAOBank::Retrieve("uniformInstance");
		shader.SetActiveShader();
		vao.Bind();
		vao.BindArrayBuffer(Bank<ArrayBuffer>::Get("aiming"), 0);
		glm::vec3 bulletPath = axes[0];
		glm::vec3 position = playerModel.translation + bulletPath * 10.f;
		Model model{ position, playerModel.rotation };
		model.scale = glm::vec3(0.5f);
		std::array<glm::mat4, 6> prospectiveBoxes{};

		for (int i = 0; i < prospectiveBoxes.size(); i++)
		{
			prospectiveBoxes[i] = model.GetModelMatrix();
			model.translation += 15.f * bulletPath;
		}
		ArrayBuffer& instances = BufferBank::Get("sleepyBoxes");
		instances.BufferData(prospectiveBoxes);
		vao.BindArrayBuffer(instances, 1);
		shader.SetVec4("Color", glm::vec4(1.f, 0.25f, 0.1f, 1.f));
		shader.DrawArrayInstanced<DrawType::Lines>(Bank<ArrayBuffer>::Get("aiming"), instances);
	}

	{
		Model defaults(playerModel);
		Shader& ship = ShaderBank::Retrieve("ship");
		ship.SetActiveShader();
		VAOBank::Retrieve("meshVertex").Bind();
		defaults.translation = glm::vec3(10, 10, 0);
		defaults.rotation = glm::quat(0.f, 0.f, 0.f, 1.f);
		defaults.scale = glm::vec3(0.5f);
		ship.SetMat4("modelMat", defaults.GetModelMatrix());
		ship.SetMat4("normalMat", defaults.GetNormalMatrix());
		groovy.Draw(ship);
	}

	{
		DisableDepthWritePush;
		EnablePushFlags(Blending);
		// TODO: Maybe look into this https://www.opengl.org/archives/resources/code/samples/sig99/advanced99/notes/node20.html
		ArrayBuffer& decals = BufferBank::Retrieve("decals");
		Shader& shader = ShaderBank::Retrieve("decalShader");
		shader.SetActiveShader();
		VAOBank::Retrieve("texturedVAO").DoubleBindArrayBuffer(decals);
		shader.SetTextureUnit("textureIn", texture, 0);
		shader.DrawArray<DrawType::Triangle>(decals);
	}
	
	trashMan.Draw(ShaderBank::Retrieve("debris"));
	if (featureToggle)
	{
		DisablePushFlags(FaceCulling);
		EnablePushFlags(Blending);
		Shader& trails = ShaderBank::Retrieve("trail");
		trails.SetActiveShader();
		VAO& colorVAO = VAOBank::Retrieve("colorVAO");

		colorVAO.Bind();
		trails.SetVec3("Color", glm::vec3(2.f, 204.f, 254.f) / 255.f);
		ArrayBuffer& leftBuffer = BufferBank::Retrieve("leftBuffer"), & rightBuffer = BufferBank::Retrieve("rightBuffer");
		colorVAO.BindArrayBuffer(leftBuffer);
		trails.DrawArray<DrawType::TriangleStrip>(leftBuffer);
		colorVAO.BindArrayBuffer(rightBuffer);
		trails.DrawArray<DrawType::TriangleStrip>(rightBuffer);
	}

	{
		Shader& engine = ShaderBank::Retrieve("engine");
		engine.SetActiveShader();
		VAOBank::Get("engineInstance").DoubleBindArrayBuffer(exhaustBuffer);
		engine.SetUnsignedInt("Time", static_cast<unsigned int>(gameTicks & std::numeric_limits<unsigned int>::max()));
		engine.SetUnsignedInt("Period", 150);
		engine.DrawArrayInstanced<DrawType::Triangle>(Bank<ArrayBuffer>::Get("dummyEngine"), exhaustBuffer);
	}
	
	if (bulletMesh.rawIndirect[1].instanceCount > 0)
	{
		Shader& bulletShader = ShaderBank::Get("bulletShader");
		VAO& vao = VAOBank::Retrieve("bulletVAO");
		bulletShader.SetActiveShader();
		bulletMesh.Bind(vao);
		vao.BindArrayBuffer(BufferBank::Retrieve("bulletMats"), 1);
		bulletShader.MultiDrawElements(bulletMesh.indirect);
	}

	// Everything potentially transparent has to be drawn *after* the skybox
	if (!debugFlags[CHECK_LIGHT_TILES])
	{
		DisablePushFlags(FaceCulling);
		DisableDepthWritePush;
		Shader& skyBox = ShaderBank::Retrieve("skyBox");
		skyBox.SetActiveShader();
		VAOBank::Retrieve("uniform").DoubleBindArrayBuffer(Bank<ArrayBuffer>::Get("plainCube"), 0);
		skyBox.SetTextureUnit("skyBox", sky);
		skyBox.DrawElements<DrawType::Triangle>(solidCubeIndex);
	}
	// Drawin quicklights
	if (quickLightPairs.ExclusiveOperation(
		[](auto& data)
		{
			if (data.size() > 0)
			{
				BufferBank::Get("Quicker").BufferData(data);
			}
			return data.size();
		}
	) > 0)
	{
		EnablePushFlags(Blending);
		Shader& shader = ShaderBank::Retrieve("laser");
		shader.SetActiveShader();
		shader.SetVec4("Color", glm::vec4(1.f, 1.f, 0.f, 1.f));
		VAO& vao = VAOBank::Get("uniform");
		ArrayBuffer& buffer = BufferBank::Get("Quicker");
		vao.Bind();
		vao.BindArrayBuffer(buffer);
		shader.DrawArray<DrawType::Lines>(buffer);
	}

	{
		EnablePushFlags(Blending);
		DisablePushFlags(FaceCulling);
		DisableDepthWritePush;

		Shader& foolish = ShaderBank::Get("Shielding");
		// Simple mesh instance only uses a single position for the instanced drawing
		VAO& vao = VAOBank::Get("simple_mesh_instance");
		ArrayBuffer& buffer = Bank<ArrayBuffer>::Get("shieldPos");
		foolish.SetActiveShader();
		vao.Bind();
		vao.BindArrayBuffer(sphereBuffer, 0);
		vao.BindArrayBuffer(buffer, 1);
		sphereIndicies.BindBuffer();
		foolish.SetTextureUnit("textureIn", buffet.GetColor(), 0);
		Model maudlin;
		maudlin.scale = glm::vec3(Bank<float>::Get("ShieldSize") + Bank<float>::Get("TickTockBrain"));
		foolish.SetMat4("modelMat", maudlin.GetModelMatrix());
		foolish.SetMat4("normalMat", glm::mat4(1.f));
		foolish.SetVec3("CameraPos", localCamera);
		foolish.SetUnsignedInt("FeatureToggle", featureToggle);
		foolish.DrawElementsInstanced<DrawType::Triangle>(sphereIndicies, buffer);
	}
	// Healthbar
	{
		DisablePushFlags(DepthTesting | FaceCulling);
		ScreenRect angus = ScreenRect::CenteredAt(glm::vec2(500.f, 800.f), glm::vec2(200.f, 50.f)),
			bogus = ScreenRect::CenteredAt(glm::vec2(500.f, 800.f), glm::vec2(190.f, 40.f));
		bogus.z *= glm::length(velocity) / 60.f;
		glm::vec4 angusColor = glm::vec4(1.f, 0.f, 0.f, 1.f);
		glm::vec4 bogusColor = glm::vec4(1.f, 1.f, 0.f, 1.f);

		//if (featureToggle)
		{
			Shader& shader = ShaderBank::Retrieve("uiRect");
			VAO& vao = VAOBank::Retrieve("particle_soup");
			shader.SetActiveShader();
			vao.Bind();
			shader.SetVec4("color", angusColor);
			shader.SetVec4("rectangle", angus);
			shader.DrawArray<DrawType::TriangleStrip>(4);

			shader.SetVec4("color", bogusColor);
			shader.SetVec4("rectangle", bogus);
			shader.DrawArray<DrawType::TriangleStrip>(4);
		}
		//else
		{
			angus.y -= 150;
			bogus.y -= 150;

			ArrayBuffer& plegm = BufferBank::Get("plegm");
			struct beta { glm::vec4 a, b; };
			std::array<beta, 2> sleepyHead{ {{angus, bogusColor}, {bogus, angusColor}} };
			plegm.BufferData(sleepyHead);
			Shader & shader = ShaderBank::Retrieve("uiRect2");
			shader.SetActiveShader();
			VAO& vao = VAOBank::Retrieve("particle_soup");
			vao.Bind();
			vao.BindArrayBuffer(plegm, 0);
			screenSpaceBuffer.SetBindingPoint(1);
			screenSpaceBuffer.BindUniform();
			shader.DrawArrayInstanced<DrawType::TriangleStrip>(BufferBank::Get("dummy"), plegm);
			
		}
	}

	// Debug Info Display
	{
		DisablePushFlags(DepthTesting);
		Shader& fontShader = ShaderBank::Retrieve("fontShader");
		fontShader.SetActiveShader();
		VAOBank::Retrieve("fontVAO").DoubleBindArrayBuffer(textBuffer);
		fontShader.SetTextureUnit("fontTexture", fonter.GetTexture(), 0);
		fontShader.DrawArray<DrawType::Triangle>(textBuffer);
	}
	ShaderBank::Retrieve("widget").SetActiveShader();
	ShaderBank::Retrieve("widget").DrawArray<DrawType::Lines>(6);
	if (vertexQueryEnable)
	{
		glEndQuery(GL_PRIMITIVES_GENERATED);
	}

	{
		if (vertexQueryEnable)
		{
			glBeginQuery(GL_PRIMITIVES_GENERATED, particleCountQuery);
		}
		//glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 1, particleCountQuery);
		Shader& shader = ShaderBank::Get("particle_soup");
		VAO& vao = VAOBank::Get("particle_soup");
		auto& drawOutParts = BufferBank::Retrieve("DrawParticles");
		auto& indirectOutParts = Bank<DrawIndirectBuffer>::Retrieve("IndirectParticles");
		shader.SetActiveShader();
		vao.Bind();
		vao.BindArrayBuffer(drawOutParts);
		shader.DrawArrayIndirect<DrawType::TriangleStrip>(indirectOutParts);
		if (vertexQueryEnable)
		{
			glEndQuery(GL_PRIMITIVES_GENERATED);
		}
	}

	// Framebuffer stuff
	glBindFramebuffer(GL_READ_FRAMEBUFFER, renderTarget.GetFrameBuffer());
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glm::ivec2 dimension2 = Window::GetSize();
	// TODO: Let render resolution and screen resolution be decoupled
	glBlitFramebuffer(0, 0, dimension2.x, dimension2.y, 0, 0, dimension2.x, dimension2.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	BindDefaultFrameBuffer();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	
	auto end = std::chrono::high_resolution_clock::now();
	displayTime = end - displayStartTime;
	displayStartTime = end;
	glEndQuery(GL_TIME_ELAPSED);
	glFrameTimeQueries.push_back(currentRenderQuery);
	timeCurrentQuery = vertexQueryEnable;
}

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
	std::erase_if(glFrameTimeQueries,
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
	static GLuint64 particleCount = 0;
	static GLuint64 vertexCount = 0;
	if (particleCountQuery == 0)
	{
		glGenQueries(1, &particleCountQuery);
		glGenQueries(1, &VertexCountQuery);
	}
	else
	{
		if (timeCurrentQuery)
		{
			glGetQueryObjectui64v(particleCountQuery, GL_QUERY_RESULT, &particleCount);
			particleCount /= 4;
			glGetQueryObjectui64v(VertexCountQuery, GL_QUERY_RESULT, &vertexCount);
			timeCurrentQuery = false;
		}
	}

	// "Proper" input handling
	// These functions should be moved to the gametick loop, don't want to over-poll the input device and get weird
	Gamepad::Update();
	Mouse::UpdateEdges();
	boardState = Input::UpdateStatus();

	Input::UIStuff();
	Input::DisplayInput();
	if (!Input::ControllerActive())
	{
		float tilt = 0.f;
		if (keyState['Q']) 
		{
			tilt += 1.f;
		}
		if (keyState['E']) 
		{
			tilt += -1.f;
		}
		targetAngles.z = tilt * 0.5f;
		glm::vec3 adjustment = (shiftHeld) ? glm::vec3(0.5f) : glm::vec3(1.f);
		boardState.heading = glm::vec4(playerSpeedControl, targetAngles * adjustment);
		boardState.fireButton = Mouse::CheckButton(Mouse::ButtonRight);
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
	}

	// Should only activate once per game tick
	if (tickHappend.test())
	{
		tickHappend.clear();
		static std::size_t idleTickCount = 0;
		idleTickCount++;
		const Model playerModel = playfield.GetModel();
		{
			glm::mat3 playerLocal = static_cast<glm::mat3>(playerModel.rotation);
			glm::vec3 forward = playerLocal[0];
			glm::vec3 left = playerLocal[2];
			left *= 0.5f;
			glm::vec3 local = playerModel.translation;
			local -= forward * 0.55f;
			glm::vec3 upSet = playerLocal[1] * 0.05f;

			leftCircle.Push(ColoredVertex{ local - left,  upSet + left * 0.2f });
			leftCircle.Push(ColoredVertex{ local - left, -upSet - left * 0.2f });
			rightCircle.Push(ColoredVertex{ local + left,  upSet - left * 0.2f });
			rightCircle.Push(ColoredVertex{ local + left, -upSet + left * 0.2f });

			BufferBank::Retrieve("leftBuffer").BufferData(leftCircle.GetLinear());
			BufferBank::Retrieve("rightBuffer").BufferData(rightCircle.GetLinear());
		}
		// Better bullet drawing
		bulletMatricies.ExclusiveOperation([&](std::vector<glm::mat4>& mats)
			{
				bulletMesh.rawIndirect[0].instanceCount = 0;
				bulletMesh.rawIndirect[1].instanceCount = static_cast<GLuint>(mats.size());
				bulletMesh.indirect.BufferSubData(bulletMesh.rawIndirect);
				BufferBank::Get("bulletMats").BufferData(mats);
			}
		);

		bulletBoxes.ExclusiveOperation([&](std::vector<glm::mat4>& mats)
			{
				Bank<ArrayBuffer>::Get("bulletBoxes").BufferData(mats);
			}
		);
		Level::GetShips().UpdateMeshes();
		shieldPos.ExclusiveOperation([&](std::vector<glm::vec3>& bufs)
			{
				Bank<ArrayBuffer>::Get("shieldPos").BufferData(bufs);
			}
		);

		bigCringe.ExclusiveOperation([](const auto& ins)
			{
				BufferBank::Get("tightBoxes").BufferData(ins);
			}
		);
	}
	Parallel::SetStatus(!keyState['P']);

	std::stringstream buffered;
	buffered << playfield.GetVelocity() << ":" << glm::length(playfield.GetVelocity());
	buffered << "\n" << playfield.GetModel().translation;
	buffered << "\nFeatureToggle: " << std::boolalpha << featureToggle;
	buffered << '\n' << glm::dot(glm::normalize(playfield.GetVelocity()), playfield.GetModel().rotation * World::Forward);
	if (debugFlags[PRIMITIVE_COUNTING])
	{
		buffered << "\n\n Extreme Performance Penalty \n\n";
		buffered << "\nActive Particles: " << particleCount;
		buffered << "\nVertex Shader Invocations : " << vertexCount;
	}
	Level::SetInterest(Level::GetShips().GetPos());
	
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
			BufferBank::Retrieve("decals").BufferData(ref, StaticDraw);
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

		if (!debugFlags[FREEZE_GAMEPLAY])
		{
			std::erase_if(zoopers, [](quickLight& zoop) {return zoop.lifeTime-- == 0; });
		}
		std::vector<glm::vec3> quicklime;
		for (const auto& z : zoopers)
		{
			quicklime.push_back(z.start);
			quicklime.push_back(z.end);
		}
		quickLightPairs.Swap(quicklime);
		// Add one every every frame, why not
		if (boardState.popcornFire && Level::GetCurrentTick() % 2 == 0)
		{
			constexpr float FireMaxLength = 250.f;
			constexpr float FireStandardDeviation = 0.1f;

			glm::mat3 playerDir(playerModel.rotation);

			float yDeviation = glm::gaussRand(0.f, FireStandardDeviation);
			float zDeviation = glm::gaussRand(0.f, FireStandardDeviation);

			const glm::vec3 rayDir   = glm::normalize(playerDir * glm::vec3(2.f, yDeviation, zDeviation));
			const glm::vec3 rayStart = playerModel.translation;

			Laser::Result out = Laser::FireLaserPlayer(Ray(rayStart, rayDir), FireMaxLength);
			{
				float currentDepth = glm::length(out.start - out.end);
				float A = glm::linearRand(2.f, currentDepth);
				float B = glm::linearRand(2.f, currentDepth);
				if (A > B)
				{
					std::swap(A, B);
				}
				float duration = glm::linearRand(20.f, Tick::PerSecond / 4.f);
				glm::vec3 pointA = rayStart + rayDir * A + glm::ballRand(0.25f);
				glm::vec3 pointB = rayStart + rayDir * B;
				zoopers.emplace_back(pointA, pointB, static_cast<std::uint32_t>(duration));

				glm::vec3 color;
				if (out.type == Laser::HitType::Terrain || out.type == Laser::HitType::Entity)
				{
					RayCollision result = out.hit.value();
					//decaymen.push_back(result.point + result.normal);
					float deviation = 0.15f;
					float basis = 1.f - deviation;
					color = glm::vec3(1.f, basis, 0.f);
					color += glm::vec3(glm::diskRand(deviation), 0.f);
				}
				if (out.type == Laser::HitType::Shield)
				{
					RayCollision result = out.hit.value();
					//decaymen.push_back(result.point);
					color = ShieldColor;
				}
				if (out.hit.has_value())
				{
					RayCollision result = out.hit.value();
					glm::vec3 placement = result.point;
					glm::vec3 smartNorm = result.normal;
					std::vector<infinite_pain> painterlys;
					for (int i = 0; i < 10; i++)
					{
						infinite_pain bonk{};
						float lifetime = glm::abs(glm::gaussRand(0.35f, 0.125f)) * 2.f;
						float size = glm::linearRand(0.1f, 0.125f);
						bonk.position = glm::vec4(placement + glm::sphericalRand(0.1f), size);

						glm::vec3 simp = glm::normalize(glm::sphericalRand(1.f) + smartNorm * 1.5f);
						glm::vec3 up = simp * glm::abs(glm::gaussRand(10.f, 2.5f));
						bonk.velocity = glm::vec4(up * 0.5f, lifetime);
						bonk.normal = glm::vec4(glm::normalize(smartNorm) * 0.125f, 1.f);
						bonk.color = glm::vec4(color, 1.f);
						painterlys.push_back(bonk);
					}
					particlesNew.Swap(painterlys);
				}
			}
		}

		// Bullet stuff;
		std::vector<glm::mat4> inactive, bulletDebugBoxes;

		std::vector<LightVolume> volumes{ constantLights };
		std::erase_if(decaymen, [](const auto& decaying) {return decaying.timeLeft == 0; });
		for (auto& decay : decaymen)
		{
			volumes.push_back(decay.Tick());
		}
		Level::GetShips().Update();
		bigCringe.Swap(Level::GetShips().GetOBBS());

		auto tmep = bobert.GetPoints(Level::GetShips().GetRawPositions());
		std::vector<glm::vec3> shieldPoses; 
		
		auto& screaming = Bank<std::vector<glm::vec3>>::Get("Spheres");
		screaming.clear();
		std::copy(tmep.begin(), tmep.end(), std::back_inserter(screaming));
		
		// This is bad and should be moved to the shield generator class
		for (glm::vec3 point : tmep)
		{
			Sphere spoke(point, Bank<float>::Get("ShieldSize"));
			if (Level::GetBulletTree().QuickTest(spoke.GetAABB()))
			{
				shieldPoses.push_back(point);
			}
			volumes.push_back({ glm::vec4(point, 20.f), glm::vec4(ShieldColor, 1.f), glm::vec4(1.f, 0.5f, 0.05f, 1.f) });
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
					bulletDebugBoxes.push_back(transformedBox.GetModelMatrix());
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
						Sphere bogus{ point, Bank<float>::Get("ShieldSize") * glm::compMax(ClockBrain::Collision.GetScale()) };
						if (flipper.Intersect(bogus, hit))
						{
							if (!(bogus.SignedDistance(segmentation.A) < 0 && bogus.SignedDistance(segmentation.B) < 0))
							{
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
								bulletDebugBoxes.push_back(sigma.GetModelMatrix());
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
		if (bulletDebugBoxes.size() > 0)
		{
			bulletBoxes.Swap(bulletDebugBoxes);
		}
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

		groovy.Update();

		managedProcess.Update();

		// End of Tick housekeeping
		tickHappend.test_and_set();
		auto tickEnd = std::chrono::steady_clock::now();
		long long tickDelta = (tickEnd - tickStart).count();

		maxTickTime = std::max(tickDelta, maxTickTime);
		averageTickTime = gameTickTime.Update(tickDelta / 1000);

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
	gameTickDone = true;
}

void window_focus_callback([[maybe_unused]] GLFWwindow* window, int focused)
{
	if (!focused)
	{
		keyStateBackup.fill(false);
	}
}

void key_callback([[maybe_unused]] GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, int mods)
{
	bool state = (action == GLFW_PRESS);
	if (state)
	{
		Input::Gamepad::Deactivate();
	}
	shiftHeld = mods & GLFW_MOD_SHIFT;

	if (key < 0)
	{
		return;
	}

	unsigned char letter = static_cast<unsigned char>(key & 0xFF);

	if (action != GLFW_RELEASE && key < 0xFF)
	{
		unsigned char copied = letter;
		if (std::isalnum(copied))
		{
			if (!(mods & GLFW_MOD_CAPS_LOCK) && mods & GLFW_MOD_SHIFT)
			{
				copied = static_cast<unsigned char>(std::tolower(copied));
			}
			else if (!(mods & GLFW_MOD_SHIFT)) 
			{
				copied = static_cast<unsigned char>(std::tolower(copied));
			}
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
		if (key == GLFW_KEY_ESCAPE) 
		{
			windowShouldClose = true;
			//glfwSetWindowShouldClose(window, GLFW_TRUE);
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

	return Ray(glm::vec3(0.f), faced);
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


	ui_tester_buffer.BufferData(NineSliceGenerate(Window::GetHalfF(), deviation), StaticDraw);
	Mouse::SetPosition(x, y);
	targetAngles = glm::vec3(0.f);

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
		Shader::ForceRecompile(true);
		Shader::PushContext();
		Shader::Define(std::format("#define SCREEN_SIZE vec2({},{})", width, height));
		Shader::Define(std::format("#define TILE_SIZE {}", static_cast<std::uint32_t>(gridResolution)));
		// Add one to account for the record keeping one
		Shader::Define(std::format("#define MASKS_PER_TILE {}", BucketsPerTile));
		ShaderBank::Get("lightCulling").CompileCompute("light_cull");
		ShaderBank::Get("computation").CompileCompute("compute_frustums");
		Shader::PopContext();
		Shader::ForceRecompile(false);

		Shader& shader = ShaderBank::Retrieve("computation");
		shader.UniformBlockBinding("Camera", 0);

		auto nextMult = [](auto a, auto b) {return glm::ceil(a / b) * b; };

		// Moving past the sample
		shader.SetActiveShader();
		// Frustum space calculations
		auto amount = nextMult(Window::GetSizeF(), gridResolution) / gridResolution;
		numTiles = static_cast<decltype(numTiles)>(amount.x * amount.y);
		tileDimension = amount;
		// The actual storage of the lights in bitmask form
		ShaderStorage::Get("LightMasks").Reserve(sizeof(std::uint32_t) * numTiles * BucketsPerTile);
		ShaderStorage::Get("LightMasks").BindBufferBase(8);
		// Frustums
		ShaderStorage::Get("Frustums").Reserve(sizeof(glm::mat4) * numTiles);
		ShaderStorage::Get("Frustums").BindBufferBase(7);

		ShaderStorage::Get("LightBlock").BufferData(std::to_array({ 0u, 0u }));
		ShaderStorage::Retrieve("LightBlock").BindBufferBase(9);
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
			uniformed.BufferSubData(glm::uvec3(BucketsPerTile, 1337, 547), 52);
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
	InitLog();
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
		Log("Failed to initialized GLFW");
		return -1;
	}
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	glfwWindowHint(GLFW_OPENGL_API, GLFW_TRUE);
	glfwWindowHint(GLFW_STEREO, GLFW_FALSE);

	glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_ANY_RELEASE_BEHAVIOR);
	glfwWindowHint(GLFW_CONTEXT_ROBUSTNESS, GLFW_LOSE_CONTEXT_ON_RESET);

#ifdef _DEBUG
	glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_FALSE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#else
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_FALSE);
#endif // _DEBUG

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

	// TODO: glfwShowWindow
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
	while (!windowShouldClose && !glfwWindowShouldClose(windowPointer))
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		idle();
		display();
		glfwSwapBuffers(windowPointer);
		glfwPollEvents();
	}
	windowShouldClose = true;
	while (!gameTickDone)
	{
		std::this_thread::yield();
	}
	glfwSetWindowShouldClose(windowPointer, GLFW_TRUE);
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	Input::Teardown();
	// TODO: cleanup
	CloseLog();
	return 0;
}

void init()
{
	TestFunc();
	//Input::ControllerStuff();
	std::srand(NULL);

	// OpenGL Feature Enabling
#ifdef _DEBUG
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(DebugCallback, nullptr);
#else
	glDisable(GL_DEBUG_OUTPUT);
#endif // _DEBUG

	EnableGLFeatures<DepthTesting | FaceCulling>();
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	DisableGLFeatures<MultiSampling>();

	DefaultDepthTest();

	glClearColor(0, 0, 0, 1);
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
	glFrontFace(GL_CCW);

	// SHADER SETUP
	Shader::SetBasePath("Shaders");
	Shader::IncludeInShaderFilesystem("lighting", "lighting.incl");
	Shader::IncludeInShaderFilesystem("camera", "camera.incl");
	Shader::IncludeInShaderFilesystem("ScreenSpace", "ScreenSpace.incl");
	Shader::IncludeInShaderFilesystem("frustums", "frustums.incl");
	Shader::IncludeInShaderFilesystem("forward_buffers", "forward_buffers.incl");
	Shader::IncludeInShaderFilesystem("forward_plus", "forward_plus.incl");
	Shader::IncludeInShaderFilesystem("cone", "cone.incl");
	Shader::IncludeInShaderFilesystem("imposter", "imposter.incl");
	Shader::IncludeInShaderFilesystem("empty", "empty.incl");
	ExternalShaders::Setup();

	{
		QUICKTIMER("Shaders");
		ShaderBank::Get("basic").CompileSingleFile("basic");
		ShaderBank::Get("bulletShader").CompileSingleFileInstanced("vertex_color");
		ShaderBank::Get("debris").CompileSingleFileInstanced("performant_mesh");
		ShaderBank::Get("skyBox").CompileSingleFile("sky");
		ShaderBank::Get("decalShader").CompileSingleFile("decal");
		ShaderBank::Get("engine").CompileSingleFile("engine");
		ShaderBank::Get("laser").CompileSingleFile("laser");
		ShaderBank::Get("laserBatch").CompileSingleFileInstanced("laser");
		ShaderBank::Get("fontShader").CompileSingleFile("font");
		ShaderBank::Get("nineSlicer").CompileSingleFile("ui_nine");
		ShaderBank::Get("trail").CompileSingleFile("trail");
		ShaderBank::Get("Shielding").CompileSingleFile("shield");
		ShaderBank::Get("ship").CompileSingleFile("performant_mesh");
		ShaderBank::Get("uiRect").CompileSingleFile("ui_rect");
		ShaderBank::Get("uiRect2").CompileSingleFileInstanced("ui_rect");
		ShaderBank::Get("uiRectTexture").CompileSingleFile("ui_rect_texture");
		ShaderBank::Get("uniform").CompileSingleFile("uniform");
		ShaderBank::Get("widget").CompileSingleFile("widget");
		ShaderBank::Get("ShieldTexture").Compile(
			"framebuffer", "shield_texture"
		);
		ShaderBank::Get("debrisCompute").CompileCompute("debris_compute");
		ShaderBank::Get("particleCompute").CompileCompute("particle_compute");
		ShaderBank::Get("depthOnly").CompileSingleFile("new_mesh_simp");
		ShaderBank::Get("forwardPlus").Compile("new_mesh", "forward_plus");
		ShaderBank::Get("forwardPlusMulti").CompileInstanced("new_mesh", "forward_plus");
		ShaderBank::Get("dust").CompileSingleFile("dust");
		ShaderBank::Get("uniformInstance").CompileSingleFileInstanced("uniform");
		ShaderBank::Get("visualize").Compile("framebuffer", "visualize");
		ShaderBank::Get("particle_soup").CompileSingleFile("particle_soup");
	}
	ShaderBank::for_each(std::to_array({ "depthOnly", "dust", "forwardPlus", "forwardPlusMulti", "engine",
		"uniformInstance", "Shielding", "debris", "bulletShader", "skyBox", "ship", "decalShader", "basic",
		"trail", "uniform", "debrisCompute"}),
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

	ShaderBank::for_each(std::to_array({ "fontShader", "uiRect", "uiRect2", "uiRectTexture", "nineSlicer"}),
		[](auto& element)
		{
			element.UniformBlockBinding("ScreenSpace", 1);
		}
	);

	// VAO SETUP
	VAOBank::Get("fontVAO").ArrayFormat<UIVertex>();

	VAOBank::Get("nineSliced").ArrayFormatOverride<glm::vec4>("rectangle", ShaderBank::Retrieve("nineSlicer"), 0, 1);

	VAOBank::Get("uniform").ArrayFormat<Vertex>();
	VAOBank::Get("meshVertex").ArrayFormat<MeshVertex>();
	VAOBank::Get("normalVertex").ArrayFormat<NormalVertex>();

	VAOBank::Get("engineInstance").ArrayFormatOverride<glm::vec4>(0, 0, 1);
	VAOBank::Get("muscle").ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, 56);
	{
		VAO& ref = VAOBank::Get("particle_soup");
		ref.ArrayFormatOverride<glm::vec4>(0, 0, 1, 0, sizeof(glm::vec4) * 2);
		ref.ArrayFormatOverride<glm::vec4>(1, 0, 1, sizeof(glm::vec4), sizeof(glm::vec4) * 2);
	}
	{
		VAO& ref = VAOBank::Get("dust");
		ref.ArrayFormatOverride<glm::vec4>(0, 0, 1);
	}
	{
		VAO& ref = VAOBank::Get("uniformInstance");
		ref.ArrayFormat<Vertex>();
		ref.ArrayFormatM<glm::mat4>(ShaderBank::Get("uniformInstance"), 1, 1, "Model");
	}
	{
		VAO& ref = VAOBank::Get("uniformInstanceSphere");
		ref.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, sizeof(MeshVertex));
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
		VAO& ref = VAOBank::Get("simple_mesh_instance");
		ref.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, sizeof(MeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(MeshVertex, normal), sizeof(MeshVertex));
		ref.ArrayFormatOverride<glm::vec2>(2, 0, 0, offsetof(MeshVertex, texture), sizeof(MeshVertex));
		ref.ArrayFormatOverride<glm::vec3>(3, 1, 1, 0, sizeof(glm::vec3));
		//ref.ArrayFormatOverride<glm::mat4>("modelMat", ShaderBank::Get("new_mesh"), 1, 1, 0, sizeof(MeshMatrix));
		//ref.ArrayFormatOverride<glm::mat4>("normalMat", ShaderBank::Get("new_mesh"), 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
	}
	{
		VAO& bulletVAO = VAOBank::Get("bulletVAO");
		bulletVAO.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0);
		bulletVAO.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(ColoredVertex, color));
		bulletVAO.ArrayFormatOverride<glm::mat4>("modelMat", ShaderBank::Get("bulletShader"), 1, 1, 0, sizeof(glm::mat4));
	}
	VAOBank::Get("texturedVAO").ArrayFormat<TextureVertex>();

	VAOBank::Get("colorVAO").ArrayFormat<ColoredVertex>();

	// TEXTURE SETUP
	// These two textures from https://opengameart.org/content/stylized-mossy-stone-pbr-texture-set, do a better credit
	Texture::SetBasePath("Textures");

	ditherTexture.Load(Dummy::dither16, InternalRed, FormatRed, DataUnsignedByte);
	ditherTexture.SetFilters(LinearLinear, MagLinear, Repeat, Repeat);

	//normalMap.Load("bear_nm.png");
	normalMap.Load("normal.png");
	normalMap.SetFilters(LinearLinear, MagLinear, MirroredRepeat, MirroredRepeat);
	normalMap.SetAnisotropy(16.f);

	//texture.Load("laserA.png");
	texture.Load("laserC.png"); // Temp switching to a properly square decal
	texture.SetFilters(LinearLinear, MagLinear, BorderClamp, BorderClamp);

	buttonB.CreateEmptyWithFilters(100, 100, InternalRGBA, {}, glm::vec4(0, 1, 1, 1));
	buttonA.CreateEmptyWithFilters(100, 100, InternalRGBA, {}, glm::vec4(1, 0.5, 1, 1));

	nineSlice.Load("9slice.png");
	nineSlice.SetFilters();
	Bank<Texture2D>::Get("flma").Load("depth.png");
	Bank<Texture2D>::Get("flma").SetFilters();

	// TODO: Use glm::noise::perlin

	Bank<ArrayBuffer>::Get("dummyEngine").BufferData(std::array<unsigned int, 36>{});

	BufferBank::Get("decals").Generate();
	stickBuffer.BufferData(Dummy::stick);
	solidCubeIndex.BufferData(Cube::GetTriangleIndex());

	Bank<ArrayBuffer>::Get("plainCube").BufferData(Cube::GetPoints());

	Bank<Texture2D>::Get("blankTexture").CreateEmpty(1, 1, InternalRGBA8, glm::vec4(1.f));
	Bank<Texture2D>::Get("blankTexture").SetFilters();

	// Cube map shenanigans
	{
		// From Here https://opengameart.org/content/space-skybox-1 under CC0 Public Domain License
		sky.Generate(std::to_array<std::string>({"skybox/space_ft.png", "skybox/space_bk.png", "skybox/space_up.png", 
			"skybox/space_dn.png", "skybox/space_rt.png", "skybox/space_lf.png"}));
	}

	Parallel::SetStatus(true);

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
		Bank<float>::Get("TickTockBrain") = glm::compMax(ClockBrain::Collision.GetScale()) * ClockBrain::GetScale();
		playerMesh = OBJReader::MeshThingy<NormalMeshVertex>("Models\\Player.glb", {}, 
			[&](auto& c) -> void
			{
				if (onlyFirst++)
					return;
				std::ranges::transform(c, std::back_inserter(painterly), [](NormalMeshVertex b) -> glm::vec3 {return b.position; });
			}
		);
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
		levelGeometry = OBJReader::MeshThingy<NormalMeshVertex>("Models\\mothership.glb",
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
	}
	Bank<ArrayBuffer>::Get("dummyInstance").BufferData(std::to_array<MeshMatrix>({ {glm::mat4(1.f), glm::mat4(1.f)} }));

	ShieldGenerator::Setup();

	BSP& bp = Bank<BSP>::Get("Fellas");
	{
		QUICKTIMER("BSP Tree");
		bp.GenerateBSP(nodeTri);
	}

	{
		for (std::size_t i = 0; i < MaxLights / 2; i++)
		{
			glm::vec3 position = glm::ballRand(zFar / 3.f);
			glm::vec3 color = glm::abs(glm::sphericalRand(1.f));
			float radius = glm::abs(glm::gaussRand(20.f, 5.f));

			// Alledgedly this is close to what unity used at one point, but  I don't know
			float quadratic = 4.f / (radius * radius);
			// TODO: Something about these lights, the constants should change based on the radius but I'm not sure how
			constantLights.push_back({ glm::vec4(position, radius),
				glm::vec4(color, 1.f), glm::vec4(1.f, 0.0f, quadratic, 1.f) });
		}
		LightVolume adequate{};
		adequate.position = glm::vec4(0.f);
		adequate.color = glm::vec4(0.5f, 0.f, 0.5f, 4.f) * 0.05f;
		adequate.direction = glm::vec4(glm::normalize(glm::vec3(0.25f, -2.f, 0.36f)), 0.f);
		adequate.constants = glm::vec4(0.f);
		constantLights.push_back(adequate);
		adequate.direction *= -1.f;
		constantLights.push_back(adequate);
		Bank<ArrayBuffer>::Get("dummy").BufferData(std::array<glm::vec3, 4>());
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
		auto& foo = Level::GetShips().Make();
		foo.Init(i > 5 ? glm::vec3(0.f, 60.f, 0.f) : glm::vec3(0.f, -60.f, 0.f));
	}
	auto& foo = Level::GetShips().Make();
	foo.Init(glm::vec3(-200.f, 60.f, 0.f));
	
	ShaderStorage::Get("RawParticles").Reserve(Max_Partilces * sizeof(infinite_pain));
	ShaderStorage::Get("DrawParticles").Reserve(sizeof(glm::vec4) * 2* Max_Partilces);
	BufferBank::Get("DrawParticles").Reserve(sizeof(glm::vec4) * 2 * Max_Partilces);
	ShaderStorage::Get("IndirectParticles").Reserve(sizeof(DrawIndirect));
	Bank<DrawIndirectBuffer>::Get("IndirectParticles").Reserve(sizeof(DrawIndirect));
	ShaderStorage::Get("MiscParticles").BufferData(std::array<unsigned int, 2>{0});
	ShaderStorage::Get("NewParticles").Reserve(0);

	// =============================================================
	{
		QuickTimer _tim{ "Sphere/Capsule Generation" };
		Sphere::GenerateMesh(sphereBuffer, sphereIndicies, 50, 50);
	}

	{
		// Targetting thing
		const float ratio = 1.f / std::sqrt(2.f);
		auto gems = std::to_array(
			{
				glm::vec3(0, -ratio, -ratio),
				glm::vec3(0, 0.f, -1.f),
				glm::vec3(0, 0.f, -1.f),
				glm::vec3(0, ratio, -ratio),

				glm::vec3(0, -ratio, ratio),
				glm::vec3(0, 0.f, 1.f),
				glm::vec3(0, 0.f, 1.f),
				glm::vec3(0, ratio, ratio),
			}
			);
		Bank<ArrayBuffer>::Get("aiming").BufferData(gems);
	}

	Font::SetFontDirectory("Fonts");
	
	DebrisManager::LoadResources();
	trashMan.Init();
	Satelite::LoadResources();

	// Awkward syntax :(
	{
		QUICKTIMER("Font Loading");
		ASCIIFont::LoadFont(fonter, "CommitMono-400-Regular.ttf", 25.f, 2, 2);
	}

	{
		ArrayBuffer& leftBuffer = BufferBank::Get("leftBuffer"), &rightBuffer = BufferBank::Get("rightBuffer");
		// TODO: proper fill of the relevant offset so there's no weird banding
		leftCircle.Fill({ playfield.GetModel().translation, glm::vec3(0.f) });
		rightCircle.Fill({ playfield.GetModel().translation, glm::vec3(0.f) });
		leftBuffer.BufferData(leftCircle.Get());
		rightBuffer.BufferData(rightCircle.Get());
		stickIndicies.BufferData(Dummy::stickDex, StaticDraw);
	}

	cubeOutlineIndex.BufferData(Cube::GetLineIndex());

	std::array<std::string, 2> buttonText{ "Soft", "Not" };

	Texture2D tempA, tempB;
	fonter.RenderToTexture(tempA, buttonText[0], glm::vec4(0, 0, 0, 1));
	fonter.RenderToTexture(tempB, buttonText[1], glm::vec4(0, 0, 0, 1));

	ColorFrameBuffer buffered;
	glm::ivec2 bufSize = glm::max(tempA.GetSize(), tempB.GetSize()) + glm::ivec2(20);
	auto sized = NineSliceGenerate(glm::ivec2(0, 0), bufSize);
	screenSpaceBuffer.Generate(StaticRead, sizeof(glm::mat4));
	screenSpaceBuffer.SetBindingPoint(1);
	screenSpaceBuffer.BindUniform();
	screenSpaceBuffer.BufferSubData(glm::ortho<float>(0, static_cast<float>(bufSize.x), static_cast<float>(bufSize.y), 0));

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
		VAOBank::Retrieve("nineSliced").Bind();
		VAOBank::Retrieve("nineSliced").BindArrayBuffer(rects);
		nineSlicer.SetTextureUnit("image", nineSlice);
		nineSlicer.DrawArrayInstanced<DrawType::TriangleStrip>(4, 9);
		uiRectTexture.SetActiveShader();
		uiRectTexture.SetVec4("rectangle", glm::vec4(0, 0, bufSize));
		uiRectTexture.SetTextureUnit("image", (j == 0) ? tempA : tempB, 0);
		uiRectTexture.DrawArray<DrawType::TriangleStrip>(4);
	}
	DisableGLFeatures<Blending>();

	playfield.sat = &groovy;
	Input::Setup();
	Log("End of Init");
}