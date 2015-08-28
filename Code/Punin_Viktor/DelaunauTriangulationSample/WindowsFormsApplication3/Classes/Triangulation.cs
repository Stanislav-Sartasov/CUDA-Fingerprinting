using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DelaunauTriangulationSample.Classes
{
    public class Triangulation
    {
        private List<Point> points;

        public List<Triangle> triangles
        {
            get;
            set;
        }
        private List<Vector> shell;

        public Triangulation(IEnumerable<Point> collection, Graphics g = null, Pen linePen = null, Pen pointPen = null, Pen newLinePen = null, int size = 0, int delay = 0)
        {
            points = new List<Point>(collection);                       //сделали копию, на всякий пожарный
            triangles = new List<Triangle>();                           //список треугольников
            shell = new List<Vector>();                                 //список векторов из границы

            Point massCenter = Point.getMassCenter(points);             //получили центр масс
            Point basePoint = massCenter.getNearestPointFrom(points);   //получили ближайшую к нему точку
            points.Remove(basePoint);                                   //убираем её из списка вершин

            List<PolarPoint> polarCoordinates = new List<PolarPoint>(); //переходим к полярным координатам
            foreach (Point point in points)                             //для каждой точки находим новые координаты
                polarCoordinates.Add(new PolarPoint(basePoint, point)); //полюс - basePoint
            polarCoordinates.Sort();                                    //сортировка, угол в приоритете

            points.Clear();                                             //больше этот список нам не пригодится
            foreach (PolarPoint polPoint in polarCoordinates)           //записываем в него координаты точек в
                points.Add(polPoint.toDecart());                        //отсортированном порядке

            Triangle t;
            for (int i = 0; i < points.Count - 1; i++)                  //ну а теперь нужно до каждой точки протянуть вектор
            {                                                           //справа и слева от него будут треугольники, а
                t = new Triangle(points[i + 1], basePoint, points[i]);  //соседнии точки образуют границу

                shell.Add(new Vector(points[i], points[i + 1]));
                triangles.Add(t);
            }


            //это необходимо для замыкания границы
            if (new Vector(basePoint, points[points.Count - 1]).getVectorMultiplication(new Vector(basePoint, points[0])) > 0)
            {
                shell.Add(new Vector(points[points.Count - 1], points[0]));
                triangles.Add(new Triangle(points[0], basePoint, points[points.Count - 1]));
            }
            else
            {
                shell.Insert(0, new Vector(basePoint, points[0]));
                shell.Add(new Vector(points[points.Count - 1], basePoint));
            }

            updateShell(); //теперь, дополним оболочку до выпуклого множества
            if (g != null)
                foreach (Triangle triangle in triangles)
                    triangle.Paint(g, linePen, pointPen, size);
            MessageBox.Show("Триангуляция построена!\nПреобразуем её к триангуляции Делоне.");

            upToDelanay(g, linePen, pointPen, newLinePen, size, delay); //преобразовываем триангуляцию к триангуляции Делоне
            if (g != null)
                foreach (Triangle triangle in triangles)
                    triangle.Paint(g, linePen, pointPen, size);

            MessageBox.Show("Готово!");
        }


        /*
         *  Суть в следующем:
         *  оболочка не выпукла, поэтому нужно достроить её до выпуклой, 
         *  все ребра в списке у нас упорядочены против хода часовой стрелки, поэтому
         *  1) берем ребро
         *  2) берем следующее
         *  3) измеряем векторное произведение, если отрицательно - идем к шагу 1, но уже с ребром из шага 2
         *     если положительно - удаляем ребро 1 и ребро 2 из оболочки, а на их место добавляем новое ребро 
         *     с началом 1, а концом 2; так же добавляем треугольник, следим за тем, чтобы он был ориентирован по часовой стрелке
         *     переходим к шагу 1, но уже с добавленным ребром
         *  ...
         *  n) останавливаемся, когда не было изменено ни одного ребра
         */
        private void updateShell()
        {
            bool shellChanged = true;
            while (shellChanged)
            {
                shellChanged = false;
                for (int i = 0; i < shell.Count - 1; i++)
                {
                    if (shell[i].getVectorMultiplication(shell[i + 1]) < 0)
                    {
                        Triangle t = new Triangle(shell[i].Start, shell[i].End, shell[i + 1].End);
                        triangles.Add(t);

                        shell[i] = new Vector(shell[i].Start, shell[i + 1].End);
                        shell.RemoveAt(i + 1);

                        shellChanged = true;
                    }
                }

                if (shell[shell.Count - 1].getVectorMultiplication(shell[0]) < 0)
                {
                    Triangle t = new Triangle(shell[shell.Count - 1].Start, shell[0].End, shell[shell.Count - 1].End);
                    triangles.Add(t);

                    shell[shell.Count - 1] = new Vector(shell[shell.Count - 1].Start, shell[0].End);
                    shell.RemoveAt(0);

                    shellChanged = true;
                }
            }
        }

        /*
         * опять таки, делаем до тех пор, пока изменен хотя бы один треугольник:
         * 0) берем треугольник(1)
         * 1) находим для него соседа для стороны a
         *      проверяем условие делоне, если не выполнено - удаляем их из списка треугольников, на их место записываем два новых
         *      переходим к шагу 0
         * 2) аналогично шагу 1, но с соседом b
         * 3) аналогично шагу 1, но с соседом c  
         */
        public void upToDelanay(Graphics g = null, Pen linePen = null, Pen pointPen = null, Pen newLinePen = null, int size = 0, int delay = 0)
        {
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

                    Triangle tleft = getNearest(now, true, false, false, copyTriangles);
                    if (tleft != null)
                        firstChanged = UpdateTriangles(now, tleft, now.a, copyTriangles, g, linePen, pointPen, newLinePen, size, delay);

                    if (firstChanged)
                    {
                        triangleChanged = true;
                        continue;
                    }

                    tleft = getNearest(now, false, true, false, copyTriangles);
                    if (tleft != null)
                        secondChanged = UpdateTriangles(now, tleft, now.b, copyTriangles, g, linePen, pointPen, newLinePen, size, delay);

                    if (secondChanged)
                    {
                        triangleChanged = true;
                        continue;
                    }

                    tleft = getNearest(now, false, false, true, copyTriangles);
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
        private Triangle getNearest(Triangle destination, bool a, bool b, bool c, List<Triangle> list)
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
        private static Point getExcessPoint(Triangle t, Section s)
        {
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
        private bool UpdateTriangles(Triangle tRight, Triangle tLeft, Section commonSection, List<Triangle> tmpResult, Graphics g = null, Pen linePen = null, Pen pointPen = null, Pen newLinePen = null, int formHeigth = 0, int delay = 0)
        {
            Point leftExcessPoint = Triangulation.getExcessPoint(tLeft, commonSection);
            Point rightExcessPoint = Triangulation.getExcessPoint(tRight, commonSection);

            if (commonSection.countAnglesSum(leftExcessPoint) + commonSection.countAnglesSum(rightExcessPoint) < 180)
            {
                Point up = commonSection.A;
                Point down = commonSection.B;
                if (new Vector(down, rightExcessPoint).getVectorMultiplication(new Vector(down, leftExcessPoint)) < 0)
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

                if (g != null)
                {
                    g.DrawLine(new Pen(Color.White, 2), (int)commonSection.A.X, formHeigth - (int)commonSection.A.Y,
                        (int)commonSection.B.X, formHeigth - (int)commonSection.B.Y);
                    g.DrawLine(newLinePen, (int)leftExcessPoint.X, formHeigth - (int)leftExcessPoint.Y, (int)rightExcessPoint.X, formHeigth - (int)rightExcessPoint.Y);

                    Thread.Sleep(delay);

                    g.DrawLine(new Pen(Color.White, 2), (int)leftExcessPoint.X, formHeigth - (int)leftExcessPoint.Y, (int)rightExcessPoint.X, formHeigth - (int)rightExcessPoint.Y);
                    g.DrawLine(linePen, (int)leftExcessPoint.X, formHeigth - (int)leftExcessPoint.Y, (int)rightExcessPoint.X, formHeigth - (int)rightExcessPoint.Y);

                    toAddFirst.Paint(g, linePen, pointPen, formHeigth);
                    toAddSecond.Paint(g, linePen, pointPen, formHeigth);
                }
                return true;
            }
            return false;
        }
    }
}
