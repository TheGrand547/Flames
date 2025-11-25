#version 440 core
#include "camera"

#include "CubeMapMath"

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;
layout(location = 3) in vec3 Position;

/* Will be put back in when moving to full models with rotation and such, or maybe not idk
layout(location = 3) in mat4 modelMat;
layout(location = 7) in mat4 normalMat;
*/

layout(location = 0) out vec4 fPos;
layout(location = 1) out vec4 fNorm;

uniform mat4 modelMat;
uniform mat4 normalMat;

layout(location = 0) uniform sampler2D textureIn;


void main()
{
	fNorm = normalMat * vec4(vNorm, 0);
	vec3 offset = vNorm * (texture(textureIn, NormToUVCubemap(vNorm)).r - 0.1f) * 0.05f;
	
	// This isn't quite right
	fPos = modelMat * vec4(vPos + offset, 1.0) + vec4(Position, 0);
	
	gl_Position = Projection * View * fPos;
}