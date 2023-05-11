//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct Ray {
	vec3 start, dir;

	Ray(vec3 start, vec3 dir) : start(start) {
		this->dir = normalize(dir);
	}
};

struct Hit {
	float t;
	vec3 position, normal;

	Hit() { this->t = -1; }
};

struct Face {
	int a, b, c;
};

struct Intersectable {
	virtual Hit intersect(const Ray& ray) = 0;
};

class Plane : public Intersectable {
	vec3 a, b, c;

public:
	Plane(const vec3& a, const vec3& b, const vec3& c): a(a), b(b), c(c) {}

	vec3 getNormal() {
		return normalize(cross(b - a, c - a));
	}

	bool isInsidePoints(const vec3& point) {
		vec3 normal = getNormal();
		return 
			dot(cross(b - a, point - a), normal) > 0 &&
			dot(cross(c - b, point - b), normal) > 0 &&
			dot(cross(a - c, point - c), normal) > 0;
	}
	
	Hit intersect(const Ray& ray) {
		Hit result;
		vec3 normal = getNormal();
		float t = dot((a - ray.start), normal) / dot(ray.dir, normal);
		result.t = t;
		result.position = ray.start + t * ray.dir;
		result.normal = normal;
		return result;
	}
};

class MoveableObject {
	vec3 position;
	float scale;
	float rotX, rotY, rotZ;

public:
	MoveableObject() {
		position = { 0, 0, 0 };
		scale = 1;
		rotX = rotY = rotZ = 0;
	}

	void moveTo(vec3 position) { this->position = position; }

	void scaleTo(float scale) { this->scale = scale; }

	void rotateX(float rot) { this->rotX = rot; }

	void rotateY(float rot) { this->rotY = rot; }

	void rotateZ(float rot) { this->rotZ = rot; }

protected:
	vec3 transformVertex(const vec3& vertex) {
		mat4 mat = 
			ScaleMatrix({ scale, scale, scale }) *
			RotationMatrix(rotX, { 1, 0, 0 }) * RotationMatrix(rotY, { 0, 1, 0 }) * RotationMatrix(rotZ, { 0, 0, 1 }) * 
			TranslateMatrix(position);
		vec4 ambient = vec4(vertex.x, vertex.y, vertex.z, 1);
		vec4 transformed = ambient * mat;
		return vec3(transformed.x, transformed.y, transformed.z);
	}

	virtual bool acceptHit(const Ray& ray, const Hit& hit) { return true; }
};

class Mesh : public MoveableObject, public Intersectable {
	std::vector<vec3> vertices;
	std::vector<Face> faces;

public:
	Mesh(std::vector<vec3> vertices, std::vector<Face> faces): vertices(vertices), faces(faces), MoveableObject() {}
	
	Hit intersect(const Ray& ray) {
		Hit result;
		for (auto& f : faces) {
			Plane p = getPlaneByFace(f);
			Hit current = p.intersect(ray);
			if (current.t > 0 && acceptHit(ray, current) && p.isInsidePoints(current.position) && (result.t < 0 || current.t < result.t))
				result = current;
		}
		return result;
	}

private:
	Plane getPlaneByFace(const Face& face) {
		return Plane(
			transformVertex(vertices.at(face.a - 1)), 
			transformVertex(vertices.at(face.b - 1)), 
			transformVertex(vertices.at(face.c - 1))
		);
	}
};

class OuterCube : public Mesh {
public:
	OuterCube(std::vector<vec3> vertices, std::vector<Face> faces): Mesh(vertices, faces) {}

protected:
	bool acceptHit(const Ray& ray, const Hit& hit) {
		return dot(ray.dir, hit.normal) > 0;
	}
};

class Bug : public Intersectable {
	vec3 position;
	vec3 normal;
	float angle;
	float height;

public:
	Bug(Hit hit, float angle, float height): angle(angle), height(height) {
		position = hit.position;
		normal = normalize(hit.normal);
	}

	void setToHit(const Hit& hit) {
		if (hit.t <= 0) return;
		position = hit.position;
		normal = normalize(hit.normal);
	}

	Hit intersect(const Ray& ray) {
		float cosAlfa = cosf(angle);
		float a = powf(dot(ray.dir, normal), 2) - dot(ray.dir, ray.dir) * powf(cosAlfa, 2);
		float b = 2 * dot(ray.dir, normal) * dot(ray.start - position, normal) - 2 * dot(ray.dir, ray.start - position) * powf(cosAlfa, 2);
		float c = powf(dot(ray.start - position, normal), 2) - dot(ray.start - position, ray.start - position) * powf(cosAlfa, 2);

		float d = b * b - 4 * a * c;

		Hit result;
		if(d >= 0) {
			float t1 = max((-b - sqrtf(d)) / (2 * a), (-b + sqrtf(d)) / (2 * a));
			float t2 = min((-b - sqrtf(d)) / (2 * a), (-b + sqrtf(d)) / (2 * a));

			if (t1 > 0) {
				vec3 hitPos = ray.start + t1 * ray.dir;
				if (dot(hitPos - position, normal) <= height && dot(hitPos - position, normal) >= 0) {
					result.t = t1;
					result.position = hitPos;
					result.normal = normalize(2 * dot(hitPos - position, normal) * normal - 2 * (hitPos - position) * powf(cosAlfa, 2));
				}
			}
			if(t2 > 0) {
				vec3 hitPos = ray.start + t2 * ray.dir;
				if (dot(hitPos - position, normal) <= height && dot(hitPos - position, normal) >= 0) {
					result.t = t2;
					result.position = hitPos;
					result.normal = normalize(2 * dot(hitPos - position, normal) * normal - 2 * (hitPos - position) * powf(cosAlfa, 2));
				}
			}
		}
		return result;
	}
};

class Camera {
	int nX, nY;
	vec3 viewer, lookAt;
	vec3 right, up;

public:
	Camera(int nX, int nY, vec3 viewer, vec3 lookAt, float fov) : nX(nX), nY(nY), viewer(viewer), lookAt(lookAt) {
		vec3 focus = lookAt - viewer;
		float distance = length(focus);
		this->right = normalize(cross(focus, vec3(0, 0, 1))) * 2 * distance * tanf(fov / 2);
		this->up = normalize(cross(this->right, focus)) * 2 * distance * tanf(fov / 2);
	}

	Ray getRay(int X, int Y) const {
		vec3 bottomLeft = lookAt - right / 2 - up / 2;
		vec3 position = bottomLeft + right * (X + 0.5f) / nX + up * (Y + 0.5f) / nY;
		return Ray(viewer, position - viewer);
	}
	
	int getWidth() const { return nX; }

	int getHeight() const { return nY; }

	vec3 getFocus() const { return lookAt - viewer; }
};

class Scene {
	Camera camera;
	std::vector<Intersectable*> objects;

public:
	Scene(Camera camera) : camera(camera) {}

	void build() {
		OuterCube* outerCube = new OuterCube(
			{
				{-0.5,  -0.5,  -0.5},
				{-0.5,  -0.5,   0.5},
				{-0.5,   0.5,  -0.5},
				{-0.5,   0.5,   0.5},
				{ 0.5,  -0.5,  -0.5},
				{ 0.5,  -0.5,   0.5},
				{ 0.5,   0.5,  -0.5},
				{ 0.5,   0.5,   0.5}
			},
			{
				{1, 7, 5},
				{1, 3, 7},
				{1, 4, 3},
				{1, 2, 4},
				{3, 8, 7},
				{3, 4, 8},
				{5, 7, 8},
				{5, 8, 6},
				{1, 5, 6},
				{1, 6, 2},
				{2, 6, 8},
				{2, 8, 4}
			}
		);

		Mesh* diamond = new Mesh(
			{
				{0.0, 0.0, 0.78},
				{0.45, 0.45, 0.0},
				{0.45, -0.45, 0.0},
				{-0.45, -0.45, 0.0},
				{-0.45, 0.45, 0.0},
				{0.0, 0.0, -0.78}
			},
			{
				{1, 2, 3},
				{1, 2, 5},
				{1, 4, 3},
				{1, 4, 5},
				{6, 2, 3},
				{6, 2, 5},
				{6, 4, 3},
				{6, 4, 5},
			}
		);

		Mesh* cube = new Mesh(
			{
				{-0.5,  -0.5,  -0.5},
				{-0.5,  -0.5,   0.5},
				{-0.5,   0.5,  -0.5},
				{-0.5,   0.5,   0.5},
				{ 0.5,  -0.5,  -0.5},
				{ 0.5,  -0.5,   0.5},
				{ 0.5,   0.5,  -0.5},
				{ 0.5,   0.5,   0.5}
			},
			{
				{1, 7, 5},
				{1, 3, 7},
				{1, 4, 3},
				{1, 2, 4},
				{3, 8, 7},
				{3, 4, 8},
				{5, 7, 8},
				{5, 8, 6},
				{1, 5, 6},
				{1, 6, 2},
				{2, 6, 8},
				{2, 8, 4}
			}
		);
		outerCube->moveTo({ 1, 0, 0 });
		outerCube->rotateZ(40 * M_PI / 180);

		diamond->scaleTo(0.4);
		diamond->moveTo({ 0.9, 0.3, -0.188 });
		diamond->rotateZ(30 * M_PI / 180);
		diamond->rotateY(-10 * M_PI / 180);

		cube->scaleTo(0.3);
		cube->moveTo({ 0.9, -0.3, -0.1 });
		cube->rotateY(20 * M_PI / 180);
		cube->rotateZ(-30 * M_PI / 180);

		objects.push_back(diamond);
		objects.push_back(outerCube);
		objects.push_back(cube);

		Hit firstHit = firstIntersect(camera.getRay(250, 300));
		Bug* firstBug = new Bug(firstHit, 25 * M_PI / 180, 0.1);
		objects.push_back(firstBug);
	}

	Hit firstIntersect(const Ray& ray) {
		Hit result;
		for (auto o : objects) {
			Hit hit = o->intersect(ray);
			if (hit.t > 0 && (result.t < 0 || hit.t < result.t))  result = hit;
		}
		if (dot(ray.dir, result.normal) > 0) result.normal = (-1) * result.normal;
		return result;
	}

	vec3 trace(const Ray& ray) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return { 0, 0, 0 };
		vec3 outRad = 0.2 * (1 - dot(hit.normal, camera.getFocus())) * vec3(1, 1, 1);
		return outRad;
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < camera.getHeight(); Y++) {
			for (int X = 0; X < camera.getWidth(); X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * camera.getWidth() + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
};

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

GPUProgram gpuProgram;

class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
Scene scene(Camera(windowWidth, windowHeight, { -1, 0, 0 }, { 0, 0, 0 }, 45 * M_PI / 180));

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
