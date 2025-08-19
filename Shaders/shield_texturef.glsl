#version 440 core
vec2 multiNoise(vec4 pos, vec4 scale, float phase, vec2 seed);
#include "hash"
#include "multiHash"
#include "interpolate"
#include "metric"
#include "noise"
#include "gradientNoise"
#include "voronoi"
#include "perlinNoise"
#include "CellularNoise"

#include "fbm"
#include "fbmImage"

/*
#include "debug"
#include "hexagons"
#include "patterns"
*/
//#include "warp"


layout(location = 0) in vec2 textureCoords;
layout(location = 0) out float fColor;

uniform float FrameTime;

void main()
{
	fColor = 1.f;
/*
	fColor = perlinNoiseWarp(textureCoords, vec2(6), 
		0.05, // Controls blob size
		1. + 2. * FrameTime, // Controls speed of development
		1.0f, // just don't touch this
		0.05, // controls blob size, again, but in a different way
		1.f); // I don't know
		*/
}