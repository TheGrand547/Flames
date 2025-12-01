#version 440 core
#include "lighting"
#include "camera"
#include "frustums"
#include "forward_buffers"
#include "cone"
#include "forward_plus"
#include "imposter"

layout(location = 0) flat in vec3 fPos;
layout(location = 1) flat in float radius;
layout(location = 2) in vec3 relativePosition;
layout(location = 3) in vec2 fTex;

layout(location = 0) out vec4 fragmentColor;

layout(location = 1) uniform vec3 shapeColor;

void main()
{
	
	vec3 cameraPos = ImposterCalculate(relativePosition, fTex, radius);
	gl_FragDepth = ImposterDepth(cameraPos);
	vec3 cameraNormal = ImposterNormal(fPos, cameraPos);

	FragData data;
	data.position = cameraPos;
	data.normal = cameraNormal;
	data.viewDirection = normalize(cameraPos);
		
	vec3 lightOut = ForwardPlusLightingViewSpace(data);
	fragmentColor = vec4(shapeColor * lightOut, 1);
}
