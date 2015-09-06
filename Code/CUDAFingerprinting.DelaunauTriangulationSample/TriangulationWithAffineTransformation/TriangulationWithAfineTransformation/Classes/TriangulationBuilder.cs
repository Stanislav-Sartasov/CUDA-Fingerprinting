using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;

namespace TriangulationWithAfineTransformation.Classes
{
    class TriangulationBuilder
    {
        private List<Point> points;

        public List<Triangle> triangles
        {
            get;
            set;
        }
        private List<Vector> shell;

        public TriangulationBuilder(IEnumerable<Point> collection, Graphics g = null, Pen linePen = null, Pen pointPen = null, Pen newLinePen = null, int formHeight = -1, int delay = -1)
        {
            points = new List<Point>(collection);                       //сделали копию, на всякий пожарный
            triangles = new List<Triangle>();                           //список треугольников
            shell = new List<Vector>();                                 //список векторов из границы

            Point massCenter = Point.GetMassCenter(points);             //получили центр масс
            Point basePoint = massCenter.GetNearestPointFrom(points);   //получили ближайшую к нему точку
            points.Remove(basePoint);                                   //убираем её из списка вершин

            List<PolarPoint> polarCoordinates = new List<PolarPoint>(); //переходим к полярным координатам
            foreach (Point point in points)                             //для каждой точки находим новые координаты
                polarCoordinates.Add(new PolarPoint(basePoint, point)); //полюс - basePoint
            polarCoordinates.Sort();                                    //сортировка, угол в приоритете

            points.Clear();                                             //больше этот список нам не пригодится
            foreach (PolarPoint polPoint in polarCoordinates)           //записываем в него координаты точек в
                points.Add(polPoint.ToCartesian());                     //отсортированном порядке

            for (int i = 0; i < points.Count - 1; i++)                  //ну а теперь нужно до каждой точки протянуть вектор
            {                                                           //справа и слева от него будут треугольники, а
                shell.Add(new Vector(points[i], points[i + 1]));        //соседние точки образуют границу(оболочку)
                triangles.Add(new Triangle(points[i + 1], basePoint, points[i]));
            }

            if (new Vector(basePoint, points[points.Count - 1]).GetVectorMultiplication(new Vector(basePoint, points[0])) > 0)  //обработка начальной точки
            {                                                                                                                   
                shell.Add(new Vector(points[points.Count - 1], points[0]));
                triangles.Add(new Triangle(points[0], basePoint, points[points.Count - 1]));
            }
            else
            {
                shell.Insert(0, new Vector(basePoint, points[0]));
                shell.Add(new Vector(points[points.Count - 1], basePoint));
            }

            UpdateShell();  //достраиваем оболочку до выпуклого множества
            if (g != null && linePen != null && newLinePen != null && formHeight != -1 && delay != -1) 
                foreach (Triangle triangle in triangles)
                    triangle.Paint(g, linePen, pointPen, formHeight);

            UpToDelanay(g, linePen, pointPen, newLinePen, formHeight, delay);   //преобразуем триангуляцию к триангуляции Делоне
            if (g != null && linePen != null && newLinePen != null && formHeight != -1 && delay != -1)
                foreach (Triangle triangle in triangles)
                    triangle.Paint(g, linePen, pointPen, formHeight);
        }
        
        private void UpdateShell()
        {
            /*
             * Идея следующая: берем два соседних ребра в оболочке, (по способу построения они идут против часовой стрелки)
             * вычисляем их векторное произведение. Если оно положительно - оболочка не выпукла, нужно добавить в неё ребро,
             * а эти два удалить. Так же, нужно добавить треугольник. И так до тех пор, пока не сделаем полный проход, при этом
             * не изменив ни одного ребра.
             */
            bool shellChanged = true;                                                                   
            while (shellChanged)
            {
                shellChanged = false;
                for (int i = 0; i < shell.Count - 1; i++)
                {
                    if (shell[i].GetVectorMultiplication(shell[i + 1]) < 0)
                    {
                        triangles.Add(new Triangle(shell[i].Start, shell[i].End, shell[i + 1].End));

                        shell[i] = new Vector(shell[i].Start, shell[i + 1].End);
                        shell.RemoveAt(i + 1);

                        shellChanged = true;
                    }
                }

                if (shell[shell.Count - 1].GetVectorMultiplication(shell[0]) < 0)
                {
                    Triangle t = new Triangle(shell[shell.Count - 1].Start,  shell[0].End, shell[shell.Count - 1].End);
                    triangles.Add(t);
                    
                    shell[shell.Count - 1] = new Vector(shell[shell.Count - 1].Start, shell[0].End);
                    shell.RemoveAt(0);

                    shellChanged = true;
                }
            }
        }
        
        public void UpToDelanay(Graphics g = null, Pen linePen = null, Pen pointPen = null, Pen newLinePen = null, int size = -1, int delay = -1)
        {
            /*
             * Идея следующая: 
             * 1) берем треугольник
             * 2) для него находим соседа для стороны a
             * 3) проверяем выполненность условия делоне (сумму углов), 
             *  если не выполнено - меняем треугольники, удаляя эти два из
             *  списка треугольников, и добавляем два новых, переходим к новой итерации цикла
             * 4) если выполнено - гоу ту (2) только с соседом для стороны б, затем для стороны с
             * 5) и так до тех пор, пока треугольники изменяются
             */
            List<Triangle> copyTriangles = new List<Triangle>(triangles);

            bool triangleChanged = true;
            while (triangleChanged)
            {
                triangleChanged = false;
                bool firstChanged = false;
                bool thirdChanged = false;
                bool secondChanged = false;

                for (int i = 0; i < copyTriangles.Count; i++)
                {
                    Triangle now = copyTriangles[i];

                    firstChanged = false;
                    thirdChanged = false;
                    secondChanged = false;

                    Triangle tleft = GetNearestTriangleWithCommonSideInList(now, true, false, false, copyTriangles);
                    if (tleft != null)
                        firstChanged = UpdateTriangles(now, tleft, now.a, copyTriangles, g, linePen, pointPen, newLinePen, size, delay);

                    if (firstChanged)
                    {
                        triangleChanged = true;
                        continue;
                    }

                    tleft = GetNearestTriangleWithCommonSideInList(now, false, true, false, copyTriangles);
                    if (tleft != null)
                        secondChanged = UpdateTriangles(now, tleft, now.b, copyTriangles, g, linePen, pointPen, newLinePen, size, delay);

                    if (secondChanged)
                    {
                        triangleChanged = true;
                        continue;
                    }

                    tleft = GetNearestTriangleWithCommonSideInList(now, false, false, true, copyTriangles);
                    if (tleft != null)
                        thirdChanged = UpdateTriangles(now, tleft, now.c, copyTriangles, g, linePen, pointPen, newLinePen, size, delay);
                    if (thirdChanged)
                    {
                        triangleChanged = true;
                        continue;
                    }
                }
            }

            triangles = copyTriangles;
        }

        private Triangle GetNearestTriangleWithCommonSideInList(Triangle destination, bool a, bool b, bool c, List<Triangle> list)
        {

            foreach (Triangle t in list)
            {
                if (t.Equals(destination))
                    continue;
                if (a && (destination.a.Equals(t.a) || destination.a.Equals(t.b) || destination.a.Equals(t.c)))
                    return t;

                if (b && (destination.b.Equals(t.a) || destination.b.Equals(t.b) || destination.b.Equals(t.c)))
                    return t;

                if (c && (destination.c.Equals(t.a) || destination.c.Equals(t.b) || destination.c.Equals(t.c)))
                    return t;

            }
            return null;
        }

        private static Point GetExcessPoint(Triangle t, Section s)
        {
            //ищем точку из треугольника, не лежащую на ребре
            Point excessPoint;
            if (!t.A.Equals(s.A) && !t.A.Equals(s.B))
                excessPoint = t.A;
            else
                if (!t.B.Equals(s.A) && !t.B.Equals(s.B))
                    excessPoint = t.B;
                else
                    excessPoint = t.C;
            return excessPoint;
        }

        private bool UpdateTriangles(Triangle tRight, Triangle tLeft, Section commonSection, List<Triangle> tmpResult, Graphics g = null, Pen linePen = null, Pen pointPen = null, Pen newLinePen = null, int formHeight = 0, int delay = 0)
        {
            //меняем общее ребро у двух треугольников
            Point leftExcessPoint = TriangulationBuilder.GetExcessPoint(tLeft, commonSection);
            Point rightExcessPoint = TriangulationBuilder.GetExcessPoint(tRight, commonSection);

            if (commonSection.CountAnglesSum(leftExcessPoint) + commonSection.CountAnglesSum(rightExcessPoint) < 180)
            {
                Point up = commonSection.A;
                Point down = commonSection.B;
                if (new Vector(down, rightExcessPoint).GetVectorMultiplication(new Vector(down, leftExcessPoint)) < 0)
                {
                    up = commonSection.B;
                    down = commonSection.A;
                }

                tmpResult.Remove(tLeft);
                tmpResult.Remove(tRight);
                Triangle toAddFirst = new Triangle(leftExcessPoint, down, rightExcessPoint);
                Triangle toAddSecond = new Triangle(rightExcessPoint, up, leftExcessPoint);
                tmpResult.Add(toAddFirst);
                tmpResult.Add(toAddSecond);

                if (g != null && linePen != null && newLinePen != null && formHeight != -1 && delay != -1)
                {
                    g.DrawLine(new Pen(Color.White, 2), (int)commonSection.A.X, formHeight - (int)commonSection.A.Y,
                        (int)commonSection.B.X, formHeight - (int)commonSection.B.Y);
                    g.DrawLine(newLinePen, (int)leftExcessPoint.X, formHeight - (int)leftExcessPoint.Y, (int)rightExcessPoint.X, formHeight - (int)rightExcessPoint.Y);
                    
                    Thread.Sleep(delay);

                    g.DrawLine(new Pen(Color.White, 2), (int)leftExcessPoint.X, formHeight - (int)leftExcessPoint.Y, (int)rightExcessPoint.X, formHeight - (int)rightExcessPoint.Y);
                    g.DrawLine(linePen, (int)leftExcessPoint.X, formHeight - (int)leftExcessPoint.Y, (int)rightExcessPoint.X, formHeight - (int)rightExcessPoint.Y);

                    toAddFirst.Paint(g, linePen, pointPen, formHeight);
                    toAddSecond.Paint(g, linePen, pointPen, formHeight);
                }
                return true;
            }
            return false;
        }   
    }
}
