#pragma once
#ifndef CUBE_H
#define CUBE_H

class Cube
{

public:
	Cube();
	Cube(Cube&& other) noexcept;
	~Cube();
};

#endif // CUBE_H
