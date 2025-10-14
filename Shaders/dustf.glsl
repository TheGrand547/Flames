#version 440 core
#include "lighting"
#include "camera"
#include "forward_buffers"
#include "forward_plus"

/*
layout(location = 0) in vec3 fPos;
layout(location = 1) in vec3 fNorm;
*/

layout(location = 0) flat in vec3 fPos;
layout(location = 1) in vec3 fNorm;
layout(location = 2) in vec3 relativePosition;
layout(location = 3) in vec2 fTex;

layout(location = 0) out vec4 fragmentColor;

layout(location = 1) uniform vec3 shapeColor;

void main()
{
	/*
	vec3 viewDirection = normalize(View[3].xyz - fPos);	
	vec3 lightOut = ForwardPlusLighting(fPos, fNorm, viewDirection);
	fragmentColor = vec4(shapeColor * lightOut, 1);
	*/
	const float radius = 0.5f;
	
	// From https://github.com/paroj/gltut/blob/master/Tut%2013%20Impostors/data/GeomImpostor.frag
	vec3 adjusted = vec3(fTex, 0.0) + relativePosition;
	vec3 ray = normalize(adjusted);
	
	float B = 2.0 * dot(ray, -relativePosition);
	float C = dot(relativePosition, relativePosition) - (radius * radius);
	
	float det = (B * B) - (4 * C);
	if(det < 0.0)
		discard;
		
	float sqrtDet = sqrt(det);
	float posT = (-B + sqrtDet)/2;
	float negT = (-B - sqrtDet)/2;
	
	float intersectT = min(posT, negT);
	
	vec3 cameraPos = ray * intersectT;
	vec4 laDeDa = Projection * vec4(cameraPos, 1.f);
	laDeDa /= laDeDa.w;
	gl_FragDepth = laDeDa.z;
	vec3 cameraNormal = normalize(cameraPos - fPos);
	
	
	vec3 viewDirection = normalize(View[3].xyz - fPos);	
	vec3 lightOut = ForwardPlusLightingViewSpace(cameraPos, cameraNormal, -normalize(cameraPos));
	fragmentColor = vec4(shapeColor * lightOut, 1);
}
