using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using System.Linq;
using System.Diagnostics;

namespace DelaunauTriangulation
{
    public partial class MainForm : Form
    {
        private static Pen defaultPointPen = new Pen(Color.Red, 2);
        private static Pen defaultLinePen = new Pen(Color.Green, 1);
        private static Pen defaultNewLintPen = new Pen(Color.Gold, 1);
        private List<Point> points = null;
        private Graphics g = null;

        public MainForm()
        {
            InitializeComponent();
            g = this.CreateGraphics();

            DelayVolume.Text = Delay.Value.ToString();
            CountVolume.Text = Count.Value.ToString();

            MessageBox.Show("Для задания точек - кликайте левой кнопкой мыши.\nЗатем правой - для построения триангуляции.");
        }

        private void MainForm_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                if (points == null)
                {
                    points = new List<Point>();
                    g.Clear(Color.White);
                }

                Point mbToAdd = new Point(e.X, Size.Height - e.Y);
                if (!points.Contains(mbToAdd))
                    points.Add(mbToAdd);
                g.DrawEllipse(new Pen(Color.Red), e.X - 1, e.Y - 1, 3, 3);
            }
            else
            {
                if (points == null)
                {
                    g.Clear(Color.White);
                    Random rnd = new Random();
                    points = new List<Point>();
                    for (int i = 0; i < Count.Value; i++)
                    {
                        Point tmp = new Point(rnd.Next(Size.Width-200) + 30 ,rnd.Next(Size.Height-100)+50);
                        if (!points.Contains(tmp))
                            points.Add(tmp);
                        g.DrawEllipse(defaultPointPen, (int)tmp.X - 1, Size.Height - (int)tmp.Y + 1, 3, 3);
                    }
                }
                Triangulation tb = new Triangulation(points, g, defaultLinePen, defaultPointPen, defaultNewLintPen, Size.Height, Delay.Value);
                points = null;
            }
        }

        private void MainForm_ResizeEnd(object sender, EventArgs e)
        {
            points = null;
            g = this.CreateGraphics();
            g.Clear(Color.White);
        }

        private void MainForm_Resize(object sender, EventArgs e)
        {
            points = null;
            g = this.CreateGraphics();
            g.Clear(Color.White);
        }

        private void Count_Scroll(object sender, EventArgs e)
        {
            CountVolume.Text = Count.Value.ToString();
        }

        private void Delay_Scroll(object sender, EventArgs e)
        {
            DelayVolume.Text = Delay.Value.ToString();
        }
    }

    public class Point
    {
        private double x;

        public double X
        {
            get
            {
                return x;
            }
        }

        private double y;

        public double Y
        {
            get
            {
                return y;
            }
        }

        public Point(double x, double y)
        {
            this.x = x;
            this.y = y;
        }

        public double getDistance(Point point)
        {
            return Math.Sqrt((x - point.x) * (x - point.x) + (y - point.y) * (y - point.y));
        }

        public Point getNearestPointFRhom(ICollection<Point> points)
        {
            var nearestPoint = 
                points.Select(x => new { Point = x, Distance = getDistance(x) }).Where(x => x.Distance != 0).OrderBy(x => x.Distance).Select(x => x.Point).FirstOrDefault();
            double nearestDistance = -1;
            Point result = null;
            foreach (Point point in points)
            {
                double distance = getDistance(point);
                if ((distance > 0) && (nearestDistance == -1 || distance < nearestDistance))
                {
                    result = point;
                    nearestDistance = distance;
                }
            }

            return result;
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            if (obj.GetType() != this.GetType())
                return false;

            return (x == ((Point)obj).x && y == ((Point)obj).y);
        }

        public override string ToString()
        {
            return "(" + x + ";" + y + ")";
        }

        public static Point getMassCenter(ICollection<Point> points)
        {
            double x = 0;
            double y = 0;

            foreach (Point point in points)
            {
                x += point.x;
                y += point.y;
            }

            x /= points.Count;
            y /= points.Count;

            return new Point(x, y);
        }

        public void Paint(Graphics g, Pen p, int formHeight) 
        {
            g.DrawEllipse(p, (int)x - 1, formHeight - (int)y + 1, 3, 3);
        }
    }

    public class Vector
    {
        private Point start;
        private Point end;

        public Point Start
        {
            get
            {
                return start;
            }
        }

        public Point End
        {
            get
            {
                return end;
            }
        }

        public Point Center
        {
            get
            {
                return new Point(start.X + 0.5 * dx, start.Y + 0.5 * dy);
            }
        }

        private double dx;
        private double dy;

        public double Dx
        {
            get
            {
                return dx;
            }
        }

        public double Dy
        {
            get
            {
                return dy;
            }
        }

        public Vector(Point start, Point end)
        {
            this.start = start;
            this.end = end;

            dx = end.X - start.X;
            dy = end.Y - Start.Y;

            length = Math.Sqrt(dx * dx + dy * dy);
        }

        private double length;

        public double Length
        {
            get
            {
                return length;
            }
        }

        public Vector getSum(Vector vector)
        {
            return new Vector(start, new Point(start.X + dx + vector.dx, start.Y + dy + vector.dy));
        }

        public double getDotMultiplication(Vector vector)
        {
            return dx * vector.dx + dy * vector.dy;
        }

        public double getVectorMultiplication(Vector vector)
        {
            return dx * vector.dy - dy * vector.dx;
        }

        public Vector getRealMultiplication(double k)
        {
            return new Vector(start, new Point(start.X + k * dx, start.Y + k * dy));
        }

        public double getSumOfAngles(Point point)
        {
            Vector vecRight = new Vector(start, point);
            Vector vecLeft = new Vector(end, point);

            double angleRight = 180 * (Math.Acos(getDotMultiplication(vecRight) / (length * vecRight.Length))) / Math.PI;
            double angleLeft = 180 * (Math.Acos((new Vector(end, start).getDotMultiplication(vecLeft)) / (length * vecLeft.Length))) / Math.PI;
            return angleLeft + angleRight;
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            if (!(obj is Vector))
                return false;

            return ((Vector)obj).Start.Equals(Start) && ((Vector)obj).End.Equals(End);
        }

        public override string ToString()
        {
            return "Vector fRhom: " + start + " to " + end;
        }
    }

    public class Section
    {
        public Vector Vector
        {
            get;
            set;
        }

        public Point A
        {
            get
            {
                return Vector.Start;
            }
        }

        public Point B
        {
            get
            {
                return Vector.End;
            }
        }

        public Triangle left
        {
            get; set;
        }

        public Triangle right
        {
            get; set;
        }

        public Section(Vector vector)
        {
            this.Vector = vector;
            left = null;
            right = null;
        }

        public Section(Vector vector, Triangle left, Triangle right)
        {
            this.Vector = vector;
            this.left = left;
            this.right = right;
        }

        public static Section getFRhom(List<Section> sections, Vector vec)
        {
            foreach (Section section in sections)
            {
                if (section.Equals(vec))
                    return section;
            }
            return null;
        }

        public double countAnglesSum(Point point)
        {
            return Vector.getSumOfAngles(point);
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;

            if (obj is Vector)
            {
                Vector vecObj = (Vector)obj;

                if (Vector.Start.Equals(vecObj.Start) && Vector.End.Equals(vecObj.End))
                    return true;
                if (vecObj.End.Equals(Vector.Start) && vecObj.Start.Equals(Vector.End))
                    return true;
                return false;
            }

            if (obj.GetType() != GetType())
                return false;

            Vector vec = ((Section)obj).Vector;
            if (Vector.Start.Equals(vec.Start) && Vector.End.Equals(vec.End))
                return true;
            if (vec.End.Equals(Vector.Start) && vec.Start.Equals(Vector.End))
                return true;
            return false;
        }
    }

    public class Triangle
    {
        public Point A
        {
            get;
            set;
        }

        public Point B
        {
            get;
            set;
        }

        public Point C
        {
            get;
            set;
        }

        public Section a
        {
            get; set;
        }

        public Section b
        {
            get; set;
        }

        public Section c
        {
            get; set;
        }

        public Triangle(Point A, Point B, Point C)
        {
            this.A = A;
            this.B = B;
            this.C = C;
            a = new Section(new Vector(B, C));
            b = new Section(new Vector(C, A));
            c = new Section(new Vector(A, B));
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            if (obj.GetType() != this.GetType())
                return false;

            Triangle tmp = (Triangle)obj;

            if (tmp.A.Equals(A) && tmp.B.Equals(B) && tmp.C.Equals(C))
                return true;
            if (tmp.A.Equals(A) && tmp.B.Equals(C) && tmp.C.Equals(B))
                return true;
            if (tmp.A.Equals(B) && tmp.B.Equals(C) && tmp.C.Equals(A))
                return true;
            if (tmp.A.Equals(B) && tmp.B.Equals(A) && tmp.C.Equals(C))
                return true;
            if (tmp.A.Equals(C) && tmp.B.Equals(B) && tmp.C.Equals(A))
                return true;
            if (tmp.A.Equals(C) && tmp.B.Equals(A) && tmp.C.Equals(B))
                return true;

            return false;
        }

        public Triangle getNearest(List<Triangle> triangles, bool a_s, bool b_s, bool c_s)
        {
            foreach (Triangle t in triangles)
                if ((t.a.Equals(a) || t.b.Equals(a) || t.c.Equals(a)) && a_s)
                    return t;
                else
                    if ((t.a.Equals(b) || t.b.Equals(b) || t.c.Equals(b)) && b_s)
                    return t;
                else
                    if ((t.a.Equals(c) || t.b.Equals(c) || t.c.Equals(c)) && c_s)
                    return t;
            return null;
        }

        public void Paint(Graphics g, Pen linePen, Pen pointPen, int formHeight) {
            g.DrawLine(linePen, (int)A.X, formHeight - (int)A.Y, (int)B.X, formHeight - (int)B.Y);
            g.DrawLine(linePen, (int)B.X, formHeight - (int)B.Y, (int)C.X, formHeight - (int)C.Y);
            g.DrawLine(linePen, (int)C.X, formHeight - (int)C.Y, (int)A.X, formHeight - (int)A.Y);
            A.Paint(g, pointPen, formHeight);
            B.Paint(g, pointPen, formHeight);
            C.Paint(g, pointPen, formHeight);
        }
    }

    public class PolarPoint : IComparable<PolarPoint>
    {
        private Point pole;

        public double cosPhi
        {
            get;
            set;
        }

        public double sinPhi
        {
            get;
            set;
        }

        public double angle
        {
            get;
            set;
        }

        private double countAngle()
        {
            //вернет угол [-pi;pi]
            
            //вычисляем арккосинусы
            double t1 = Math.Acos(cosPhi);                              
            double t2 = - t1;                                           

            //вычисляем арксинусы
            double t3 = Math.Asin(sinPhi);
            double t4;
            if (t3 > 0)
                t4 = Math.PI - t3;
            else
                t4 = -Math.PI - t3;

            if (Math.Abs(t1 - t3) < 0.01 || Math.Abs(t2 - t3) < 0.01)
                return t3;
            else
                return t4;
        }

        public double Rho
        {
            get;
            set;
        }

        public PolarPoint(Point pole, Point thisPoint)
        {
            this.pole = pole;
            Rho = pole.getDistance(thisPoint);
            cosPhi = (thisPoint.X - pole.X) / Rho;
            sinPhi = (thisPoint.Y - pole.Y) / Rho;
            angle = countAngle();
        }

        public Point toDecart()
        {
            return new Point(pole.X + Rho * cosPhi, pole.Y + Rho * sinPhi);
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            
            if (obj.GetType() != this.GetType())
                return false;

            PolarPoint polPoint = (PolarPoint)obj;
            if (polPoint.pole.Equals(pole) && polPoint.Rho == Rho && polPoint.cosPhi == cosPhi && polPoint.sinPhi == sinPhi)
                return true;
            else
                return false;
        }

        public int CompareTo(PolarPoint other)
        {
            if (angle > other.angle)
                return 1;
            if (angle == other.angle && Rho > other.Rho)
                return 1;
            if (angle == other.angle && Rho == other.Rho)
                return 0;
            return -1;
        }
    }

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
            Point basePoint = massCenter.getNearestPointFRhom(points);   //получили ближайшую к нему точку
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

            updateShell();
            if (g != null)
                foreach (Triangle triangle in triangles)
                    triangle.Paint(g, linePen, pointPen, size);
            MessageBox.Show("Триангуляция построена!\nПреобразуем её к триангуляции Делоне.");

            upToDelanay(g, linePen, pointPen, newLinePen, size, delay);
            if (g != null)
                foreach (Triangle triangle in triangles)
                    triangle.Paint(g, linePen, pointPen, size);

            MessageBox.Show("Готово!");
        }
        
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
                    Triangle t = new Triangle(shell[shell.Count - 1].Start,  shell[0].End, shell[shell.Count - 1].End);
                    triangles.Add(t);
                    
                    shell[shell.Count - 1] = new Vector(shell[shell.Count - 1].Start, shell[0].End);
                    shell.RemoveAt(0);

                    shellChanged = true;
                }
            }
        }
        
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
