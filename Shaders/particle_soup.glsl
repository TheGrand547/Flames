#version 440 core
#include "camera"


#ifdef VERTEX

layout(location = 0) in vec4 vPos;
layout(location = 1) in vec4 vColor;

layout(location = 0) flat out vec3 fPos;
layout(location = 1) flat out float radius;
layout(location = 2) out vec3 relativePosition;
layout(location = 3) out vec2 fTex;
layout(location = 4) flat out vec4 fColor;

vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};

void main()
{
	radius = vPos.w;
	vec3 pos = vPos.xyz;
	// Multiplying fTex by higher ratios and not adjusting it later leads to some weird stuff
	fTex = positions[gl_VertexID % 4].xy * radius * 1.5f;
	
	vec3 adjusted = vec3(fTex, 0) + (View * vec4(pos, 1)).xyz;
	gl_Position = Projection * vec4(adjusted, 1);
	relativePosition = (View * vec4(pos, 1)).xyz;
	fColor = vColor;
}
#endif // VERTEX

#ifdef FRAGMENT
#include "imposter"

layout(location = 0) flat in vec3 fPos;
layout(location = 1) flat in float radius;
layout(location = 2) in vec3 relativePosition;
layout(location = 3) in vec2 fTex;
layout(location = 4) flat in vec4 fColor;

layout(location = 0) out vec4 fragmentColor;

void main()
{
	vec3 cameraPos = ImposterCalculate(relativePosition, fTex, radius);
	gl_FragDepth = ImposterDepth(cameraPos);
	fragmentColor = fColor;
}
#endif // FRAGMENT