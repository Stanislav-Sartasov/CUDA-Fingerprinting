#ifndef TrianglesMatcherWithCUDAHEADER
#define TrianglesMatcherWithCUDAHEADER

#include "cuda_runtime.h"
#include "DEVICE_launch_parameters.h"

typedef struct { float x; float y; } Point;
typedef struct { Point A; Point B; Point C; } Triangle;
typedef struct { float dx; float dy; float sin_phi; float cos_phi; } Transformation;
typedef struct { Transformation transformation; float distance; } TransformationWithDistance;
typedef enum{ TYPE_ABC, TYPE_BCA, TYPE_CAB }TriangleType;

/*
	Attention!
	Ориентация треугольников должна совпадать! - либо оба по часовой стрелке, либо оба против! Это действительно важно!
	В рамках данной задачи, необходимо найти такое преобразование (перенос + поворот), которое будет
	минимизировать расстояние между вершинами треугольников ABC(к нему будем стремится)
	и ABC_ - его будем крутить, вертеть, параллельно сдвигать.

	Заметка: сначала составил функцию. Она зависела от трех переменных(dx, dy, dphi). Далее нашел дифференциал, приравнял его к нулю.
	Получилась нелинейная система из трех уравнений, с тремя неизвестными. Чего же тут думать? Составил якобиан, и решал как обычную нелинейную систему.
	Была огромнейшая погрешность. Приходилось брать около 360 начальных условий, чтобы ответ был более-менее точным. Теперь вопрос - нельзя ли проще?
	Ведь можно 360 раз просто повращать треугольник, а потом сделать начальный перенос?
	Уже лучше. Итераций меньше, ошибка мала. Но все же 360 итераций... Приглядевший к уравнениям в нелинейной системе, я решил поступить следующим образом:

	Как мы будем преобразовывать треугольник ABC_? Сначала сдвинем его центр масс в начало координат, затем повернем на необходимый угол,
	а после перенесем его центр масс в центр масс треугольника ABC. Что нам это дает?
	1) вместо функции от трех переменных будем рассматривать лишь фукнцию от phi
	2) сходимость гораздо быстрее, чем у нелинейной системы, да и на порядок меньше чем у предыдущего алгоритма (~10-15 итераций, не более)

	Предложен следующий алгоритм:
	0) смещаем центры масс треугольников, в результирующем преобразовании: dx = -ABCmc.x, dy = -ABCmc.y, где ABCmc - центр масс треугольника ABC
	1) обозначим искомый треугольник ABC~
	2) тогда функция расстояния запишется следующим образом:
		F = (A-A~)^2 + (B-B~)^2 + (C-C~)^2 =
		= (Ax - A~x)^2 + (Ay - A~y)^2 + ... + (Cx - C~x)^2 + (Cy - C~y)^2,
		при этом:
		A~x = A_x cos_phi - A_y sin_phi
		A~y = A_x sin_phi + A_y cos_phi
	3) нужно найти минимум, поэтому необходимо найти нули производной
		dF = 2 * ( sin_phi(Ax * A_x + Ay * A_y) + cos_phi(Ax * A_y - Ay * A_x) + ...)
		dF^2 = 2 * ( cos_phi(Ax * A_x + Ay * A_y) - sin_phi(Ax * A_y - Ay * A_x)  + ...)

	4) обозначим sProd (скалярные произведения) = Ax * A_x + Ay * A_y + ... + Cx * C_x + Cy * C_y
				 vProd (векторные произведения) = Ax * A_y - Ay * A_x + ... + Cx * C_y - Cy * C_x

		-> dF/2     = sin_phi * sProd + cos_phi * vProd
		-> (dF^2)/2 = cos_phi * sProd - sin_phi * vProd

	5) решаем это нелинейное уравнение с несколькими начальными приближениями, получаем искомый угол
	6) в результирующее расстояние записываем косинус этого угла, а так же синус, измеряем расстояние между соответсвущими вершинами

	7) повторяем шаги 3-6 для треугольников BCA_, CAB_
	8) находим преобразование с минимальным расстоянием, оно и будет искомым
	9) profit!

	Заметка:
	рекомендуется вызвать функцию
	findOptimumTransformationPlusDistanceVersionCUDA(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, Transformation* result, float* resultDistance, int maxIterations, float e, int parts)
		ABC_   - массив треугольников, которые нужно сдвинуть,   ABC_size - его размер
		ABC    - массив треугольников, к которым нужно сдвинуть, ABCsize  - его размер
		result - массив преобразований, размером ABC_size * ABCsize, память под него нужно освободить заранее(!)
		resultDistance - массив дистанций, аналогично предыдущему
		maxIteration   - максимальное число итераций для нахождения оптимального угла       рекомендуется ~ 10-15
		e      - допустимая погрешность														рекомендуется ~ 0.00001f - 0.000001f
		parts  - число начальных приближений для угла phi									рекомендуется ~ 3-5
*/


/*
Вычисление расстояния между двумя точками, без извлечения корня.
*/
__host__ __device__ float countDistanceBetweenPoints(Point first, Point second);
/*
Вычисление образа точки, сдвинутой относительно исходной на dx, dy
*/
__host__ __device__ Point countMovedPoint(Point p, float dx, float dy);
/*
Вычисление образа точки, после поворота относительно начала координат на угол phi,
нужно передать его синус и косинус
*/
__host__ __device__ Point countRotatedPoint(Point p, float cos_phi, float sin_phi);


//треугольники

/*
Вычисление суммы расстояний между соответсвующими вершинами двух треугольников
*/
__host__ __device__ float countDistanceBetweenTriangles(Triangle* first, Triangle* second);
/*
Вычисление центра масс
*/
__host__ __device__ Point countTriangleMassCenter(Triangle* ABC);
/*
Вычисление образа треугольника, передвинутого на dx, dy
*/
__host__ __device__ Triangle countMovedTriangle(Triangle* ABC, float dx, float dy);
/*
Вычисление образа треугольника, повернутого относительно начала координат на угол phi
необходимо передать его синус и косинус
*/
__host__ __device__ Triangle countRotatedTriangle(Triangle* ABC, float cos_phi, float sin_phi);
/*
1)сдвигаем треугольник в начало координат
2)поворачиваем
3)передвигаем на dx, dy
*/
__host__ __device__ Triangle countTransformedTriangle(Triangle* ABC, Transformation t);


//преобразования
/*
Находим оптимальное преобазование среди ABC_ -> ABC, BCA_ -> ABC, CBA_ -> ABC
*/
__host__ __device__ TransformationWithDistance findOptimumTransformation(Triangle* ABC_, Triangle* ABC, float e, int maxIterations, int parts);

cudaError_t findOptimumTransformationNonParallelForWithCuda(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts);
cudaError_t findOptimumTransformationParallelForWithCuda(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts);
cudaError_t findOptimumTransformationPlusDistanceVersionCUDA(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, Transformation* result, float* resultDistance, int maxIterations, float e, int parts);
#endif