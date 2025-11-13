#version 440 core
#include "lighting"
#include "camera"
#include "frustums"
#include "forward_buffers"
#include "cone"
#include "forward_plus"

layout(location = 0) flat in vec3 fPos;
layout(location = 1) in float radius;
layout(location = 2) in vec3 relativePosition;
layout(location = 3) in vec2 fTex;

layout(location = 0) out vec4 fragmentColor;

layout(location = 1) uniform vec3 shapeColor;

float CameraToDepth(float depth)
{
	mat2 bottom = mat2(vec2(Projection[2][2], Projection[2][3]),
					vec2(Projection[3][2], Projection[3][3]));
	vec2 temp = bottom * vec2(depth, 1.f);
	return temp.x / temp.y;
}

uniform vec3 lightForward;
uniform vec3 lightPosition;

void main()
{
	// From https://github.com/paroj/gltut/blob/master/Tut%2013%20Impostors/data/GeomImpostor.frag
	vec3 adjusted = vec3(fTex, 0.0) + relativePosition;
	vec3 ray = normalize(adjusted);
	
	float B = 2.0 * dot(ray, -relativePosition);
	float C = dot(relativePosition, relativePosition) - (radius * radius);
	
	float det = (B * B) - (4 * C);
	if(det < 0.0)
		discard;
		
	float sqrtDet = sqrt(det);
	float posT = (-B + sqrtDet) / 2;
	float negT = (-B - sqrtDet) / 2;
	
	float intersectT = min(posT, negT);
	
	vec3 cameraPos = ray * intersectT;
	gl_FragDepth = CameraToDepth(cameraPos.z);
	vec3 cameraNormal = normalize(cameraPos - fPos);
	
	const vec3 FlashLightColor = vec3(148, 252, 255) / 255;
	
	vec3 viewDirection = normalize(View[3].xyz - fPos);	
	vec3 lightOut = ForwardPlusLightingViewSpace(cameraPos, cameraNormal, -normalize(cameraPos));
	//lightOut += DirectedPointLight(lightPosition, lightForward, FlashLightColor, cameraNormal, cameraPos, -normalize(cameraPos));
	fragmentColor = vec4(shapeColor * lightOut, 1);
}
