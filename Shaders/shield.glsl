#version 440 core
#include "camera"
#include "CubeMapMath"


#ifdef VERTEX

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;
layout(location = 3) in vec3 Position;

#ifdef INSTANCED

layout(location = 3) in mat4 modelMat;
layout(location = 7) in mat4 normalMat;

#else // INSTANCED

uniform mat4 modelMat;
uniform mat4 normalMat;

#endif // INSTANCED

layout(location = 0) out vec4 fPos;
layout(location = 1) out vec4 fNorm;

layout(location = 0) uniform sampler2D textureIn;


void main()
{
	fNorm = normalMat * vec4(vNorm, 0);
	vec3 offset = vNorm * (texture(textureIn, NormToUVCubemap(vNorm)).r - 0.1f) * 0.05f;
	
	// This isn't quite right
	fPos = modelMat * vec4(vPos + offset, 1.0) + vec4(Position, 0);
	
	gl_Position = Projection * View * fPos;
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) out vec4 colorOut;

layout(location = 0) in vec4 fPos;
layout(location = 1) in vec4 fNorm;

uniform sampler2D textureIn;
uniform vec3 CameraPos;

void main()
{
	const vec4 ShieldColor = vec4(120,204,226, 255) / 255;
	colorOut = ShieldColor * texture(textureIn, NormToUVCubemap(fNorm.xyz)).r;
	
	const vec3 viewDirection = normalize(CameraPos - fPos.xyz);
	
	float alignment = 1.f - abs(dot(viewDirection, normalize(fNorm.xyz)));
	alignment = pow(alignment, 2);
	colorOut.w *= smoothstep(0.f, colorOut.w, alignment);

}

#endif // FRAGMENT