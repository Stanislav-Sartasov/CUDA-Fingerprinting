using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OptimumConversionOfTriangles
{
    public partial class MainForm : Form
    {
        private static Point A = null;
        private static Point B = null;
        private static Point C = null;
        private static Point A_ = null;
        private static Point B_ = null;
        private static Point C_ = null;

        private static Pen APen = new Pen(Color.Red, 2);
        private static Pen BPen = new Pen(Color.Green, 2);
        private static Pen CPen = new Pen(Color.Blue, 2);

        private static Pen defaultLinePen = new Pen(Color.OrangeRed, 1);

        private static int formHeight = 0;

        private static Graphics g = null;

        public MainForm()
        {
            InitializeComponent();
            g = this.CreateGraphics();
            formHeight = Size.Height;
        }

        private void MainForm_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == System.Windows.Forms.MouseButtons.Right)
                g.Clear(Color.White);
            else
            if (A == null) {
                g.Clear(Color.White);
                A = new Point(e.X, formHeight - e.Y);
                A.Paint(g, APen, formHeight);
            }
            else
            if (B == null)
            {
                B = new Point(e.X, formHeight - e.Y);
                B.Paint(g, BPen, formHeight);
                g.DrawLine(defaultLinePen, (int)A.X, formHeight - (int)A.Y, (int)B.X, formHeight - (int)B.Y);
            }
            else
            if(C == null)
            {
                C = new Point(e.X, formHeight - e.Y);
                C.Paint(g, CPen, formHeight);
                g.DrawLine(defaultLinePen, (int)C.X, formHeight - (int)C.Y, (int)B.X, formHeight - (int)B.Y);
                g.DrawLine(defaultLinePen, (int)A.X, formHeight - (int)A.Y, (int)C.X, formHeight - (int)C.Y);
            }
            else
            if (A_ == null)
            {
                A_ = new Point(e.X, formHeight - e.Y);
                A_.Paint(g, APen, formHeight);
            }
            else
            if (B_ == null)
            {
                B_ = new Point(e.X, formHeight - e.Y);
                B_.Paint(g, BPen, formHeight);
                g.DrawLine(defaultLinePen, (int)A_.X, formHeight - (int)A_.Y, (int)B_.X, formHeight - (int)B_.Y);
            }
            else
                if (C_ == null)
                {
                    C_ = new Point(e.X, formHeight - e.Y);
                    C_.Paint(g, CPen, formHeight);
                    g.DrawLine(defaultLinePen, (int)C_.X, formHeight - (int)C_.Y, (int)B_.X, formHeight - (int)B_.Y);
                    g.DrawLine(defaultLinePen, (int)A_.X, formHeight - (int)A_.Y, (int)C_.X, formHeight - (int)C_.Y);
                }
                else
                {
                    Triangle ABC = new Triangle(A, B, C);

                    Triangle ABC_ = new Triangle(A_, B_, C_);

                    //ConversionOptimizator co = new ConversionOptimizator(ABC_, ABC);
                    //Triangle result = co.result;
                    ConversionOperator co = new ConversionOperator(ABC_, ABC, !CanRenamePoints.Checked);
                    Triangle result = co.result;
                    
                    result.Paint(g, new Pen(Color.Gold, 3), new Pen(Color.Red, 3), formHeight);
                    result.A.Paint(g, APen, formHeight);
                    result.B.Paint(g, BPen, formHeight);
                    result.C.Paint(g, CPen, formHeight);

                    A = null;
                    B = null;
                    C = null;
                    A_ = null;
                    B_ = null;
                    C_ = null;
                }
        }

        public void OnChangeSize(object sender, EventArgs e)
        {
            g = this.CreateGraphics();
            g.Clear(Color.White);
            A = null;
            B = null;
            C = null;
            A_ = null;
            B_ = null;
            C_ = null;
        }
    }

    public static class Matrix{
        public static double[] SolveLinearSystem(double[,] matrix, double[] result)
        {
            double[] x = new double[result.Length];

            for (int i = 0; i < result.Length; i++)
            {
                for (int j = i; j < result.Length; j++)
                    if (Math.Abs(matrix[j, i]) > Math.Abs(matrix[i, i]))
                    {
                        double tmp;
                        for (int k = i; k < result.Length; k++)
                        {
                            tmp = matrix[i, k];
                            matrix[i, k] = matrix[j, k];
                            matrix[j, k] = tmp;
                        }
                        tmp = result[i];
                        result[i] = result[j];
                        result[j] = tmp;
                    }
                for (int j = i + 1; j < result.Length; j++)
                {
                    double koef = matrix[j, i] / matrix[i, i];
                    for (int k = i; k < result.Length; k++)
                        matrix[j, k] -= matrix[i, k] * koef;
                    result[j] -= result[i] * koef;
                }
            }
            for (int i = result.Length - 1; i > -1; i--)
            {
                double sum = 0;
                for (int k = result.Length - 1; k > i; k--)
                    sum += matrix[i, k] * x[k];
                x[i] = (result[i] - sum) / matrix[i, i];
            }
            return x;
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
        
        public static Point getMassCenter(Point A, Point B, Point C){
            return new Point((A.x + B.x + C.x)/3,(A.y + B.y + C.y)/3);
        }

        public void Paint(Graphics g, Pen p, int formHeight = 0)
        {
            if (formHeight != 0)
                g.DrawEllipse(p, (int)x - 1, formHeight - (int)y - 1, 3, 3);
            else
                g.DrawEllipse(p, (int)x - 1, (int)y - 1, 3, 3);
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
            private set;
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
            get;
            set;
        }

        public Triangle right
        {
            get;
            set;
        }

        public double Length 
        {
            get { return Vector.Length; }
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
            get;
            set;
        }

        public Section b
        {
            get;
            set;
        }

        public Section c
        {
            get;
            set;
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

        public void Paint(Graphics g, Pen linePen, Pen pointPen, int formHeight)
        {
            g.DrawLine(linePen, (int)A.X, formHeight - (int)A.Y, (int)B.X, formHeight - (int)B.Y);
            g.DrawLine(linePen, (int)B.X, formHeight - (int)B.Y, (int)C.X, formHeight - (int)C.Y);
            g.DrawLine(linePen, (int)C.X, formHeight - (int)C.Y, (int)A.X, formHeight - (int)A.Y);
            A.Paint(g, pointPen, formHeight);
            B.Paint(g, pointPen, formHeight);
            C.Paint(g, pointPen, formHeight);
        }

        public Triangle getTransformation(double dx, double dy, double phi) {
            Point A__ = new Point(A.X * Math.Cos(phi) - A.Y * Math.Sin(phi) + dx,
                A.X * Math.Sin(phi) + A.Y * Math.Cos(phi) + dy);
            
            Point B__ = new Point(B.X * Math.Cos(phi) - B.Y * Math.Sin(phi) + dx,
                B.X * Math.Sin(phi) + B.Y * Math.Cos(phi) + dy);

            Point C__ = new Point(C.X * Math.Cos(phi) - C.Y * Math.Sin(phi) + dx,
                C.X * Math.Sin(phi) + C.Y * Math.Cos(phi) + dy);

            return new Triangle(A__, B__, C__);
        }
        
        public double getDistanceTo(Triangle triangle) 
        {
            return triangle.A.getDistance(A) * triangle.A.getDistance(A) +
                (triangle.B.getDistance(B) * triangle.B.getDistance(B)) +
                (triangle.C.getDistance(C) * triangle.C.getDistance(C));
        }
    }

    public class ConversionOperator {

        public Triangle result
        {
            get;
            private set;
        }

        private double[] transformation;

        public double dx { get {return transformation[0]; } }
        public double dy { get { return transformation[1]; } }
        public double phi { get { return transformation[2]; } }

        public ConversionOperator(Triangle src, Triangle dest, bool canChangeVertexes = false) 
        {
            result = src;
            transformation = new double[3];
            double distanceOld = src.getDistanceTo(dest);

            int n = 180;
            double nearestDistance = distanceOld;
            for (int i = 0; i < n; i++)
            {
                double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                OptimumConversionOperator oco = new OptimumConversionOperator(src.getTransformation(0,0,phi), dest);
                Triangle tmpResult = oco.getOptimumTriangle();
                double distanceNew = dest.getDistanceTo(tmpResult);
                if (distanceNew < nearestDistance)
                {
                    nearestDistance = distanceNew;
                    result = tmpResult;
                    transformation = oco.transformation;
                }
            }

            Triangle src_change = new Triangle(src.A, src.C, src.B);
            for (int i = 0; i < n; i++)
            {
                double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                OptimumConversionOperator oco = new OptimumConversionOperator(src_change.getTransformation(0, 0, phi), dest);
                Triangle tmpResult = oco.getOptimumTriangle();
                double distance = dest.getDistanceTo(tmpResult);
                if (nearestDistance == -1 || distance < nearestDistance)
                {
                    nearestDistance = distance;
                    result = tmpResult;
                    transformation = oco.transformation;
                }
            }

            if (canChangeVertexes)
            {
                src_change = new Triangle(src.B, src.C, src.A);
                for (int i = 0; i < n; i++)
                {
                    double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                    OptimumConversionOperator oco = new OptimumConversionOperator(src_change.getTransformation(0, 0, phi), dest);
                    Triangle tmpResult = oco.getOptimumTriangle();
                    double distance = dest.getDistanceTo(tmpResult);
                    if (nearestDistance == -1 || distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        result = tmpResult;
                        transformation = oco.transformation;
                    }
                }

                src_change = new Triangle(src.B, src.A, src.C);
                for (int i = 0; i < n; i++)
                {
                    double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                    OptimumConversionOperator oco = new OptimumConversionOperator(src_change.getTransformation(0, 0, phi), dest);
                    Triangle tmpResult = oco.getOptimumTriangle();
                    double distance = dest.getDistanceTo(tmpResult);
                    if (nearestDistance == -1 || distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        result = tmpResult;
                        transformation = oco.transformation;
                    }
                }

                src_change = new Triangle(src.C, src.A, src.B);
                for (int i = 0; i < n; i++)
                {
                    double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                    OptimumConversionOperator oco = new OptimumConversionOperator(src_change.getTransformation(0, 0, phi), dest);
                    Triangle tmpResult = oco.getOptimumTriangle();
                    double distance = dest.getDistanceTo(tmpResult);
                    if (nearestDistance == -1 || distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        result = tmpResult;
                        transformation = oco.transformation;
                    }
                }

                src_change = new Triangle(src.C, src.B, src.A);
                for (int i = 0; i < n; i++)
                {
                    double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                    OptimumConversionOperator oco = new OptimumConversionOperator(src_change.getTransformation(0, 0, phi), dest);
                    Triangle tmpResult = oco.getOptimumTriangle();
                    double distance = dest.getDistanceTo(tmpResult);
                    if (nearestDistance == -1 || distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        result = tmpResult;
                        transformation = oco.transformation;
                    }
                }
            }

        }
        
        public class OptimumConversionOperator
        {
            public Triangle ABC
            {
                get;
                private set;
            }
            private double ABCx;
            private double ABCy;

            public Triangle ABC_
            {
                get;
                private set;
            }
            private double ABC_x;
            private double ABC_y;

            public double[] transformation
            {
                get;
                private set;
            }

            private double vectorsProductions;
            private double scalarProductions;

            public OptimumConversionOperator(Triangle src, Triangle dest)
            {
                ABC = dest;
                Point ABCcenter = Point.getMassCenter(dest.A, dest.B, dest.C);
                ABCx = ABCcenter.X;
                ABCy = ABCcenter.Y;

                ABC_ = src;
                Point ABC_center = Point.getMassCenter(src.A, src.B, src.C);
                ABC_x = ABC_center.X;
                ABC_y = ABC_center.Y;

                vectorsProductions = ABC.A.X * ABC_.A.Y - ABC.A.Y * ABC_.A.X +
                        +ABC.B.X * ABC_.B.Y - ABC.B.Y * ABC_.B.X +
                        +ABC.C.X * ABC_.C.Y - ABC.C.Y * ABC_.C.X;

                scalarProductions = ABC.A.X * ABC_.A.X + ABC.A.Y * ABC_.A.Y +
                        +ABC.B.X * ABC_.B.X + ABC.B.Y * ABC_.B.Y +
                        +ABC.C.X * ABC_.C.X + ABC.C.Y * ABC_.C.Y;
            }
            public Triangle getOptimumTriangle()
            {
                transformation = SolveNonLinearSystem(0.0000000001, 10000);

                return ABC_.getTransformation(transformation[0], transformation[1], transformation[2]);
            }

            private double f1(double dx, double dy, double phi)
            {
                return ABC_x * Math.Cos(phi) - ABC_y * Math.Sin(phi) + dx - ABCx;
            }
            private double f2(double dx, double dy, double phi)
            {
                return ABC_x * Math.Sin(phi) + ABC_y * Math.Cos(phi) + dy - ABCy;
            }
            private double f3(double dx, double dy, double phi)
            {
                return Math.Cos(phi) * (3 * dx * ABC_y - 3 * dy * ABC_x + vectorsProductions) +
                    Math.Sin(phi) * (3 * dx * ABC_x + 3 * dy * ABC_y - scalarProductions);
            }

            private double df1dx(double dx, double dy, double phi)
            {
                return 1;
            }
            private double df1dy(double dx, double dy, double phi)
            {
                return 0;
            }
            private double df1dphi(double dx, double dy, double phi)
            {
                return -ABC_x * Math.Sin(phi) - ABC_y * Math.Cos(phi);
            }

            private double df2dx(double dx, double dy, double phi)
            {
                return 0;
            }
            private double df2dy(double dx, double dy, double phi)
            {
                return 1;
            }
            private double df2dphi(double dx, double dy, double phi)
            {
                return ABC_x * Math.Cos(phi) - ABC_y * Math.Sin(phi);
            }

            private double df3dx(double dx, double dy, double phi)
            {
                return 3 * ABC_y * Math.Cos(phi) + 3 * ABC_x * Math.Sin(phi);
            }
            private double df3dy(double dx, double dy, double phi)
            {
                return -3 * ABC_x * Math.Cos(phi) - 3 * ABC_y * Math.Sin(phi);
            }
            private double df3dphi(double dx, double dy, double phi)
            {
                return -Math.Sin(phi) * (3 * dx * ABC_y - 3 * dy * ABC_x + vectorsProductions) +
                        Math.Cos(phi) * (3 * dx * ABC_x + 3 * dy * ABC_y - scalarProductions);
            }

            private double[,] getJacobian(double dx, double dy, double phi)
            {
                double[,] matrix = new double[3, 3];
                matrix[0, 0] = df1dx(dx, dy, phi);
                matrix[0, 1] = df1dy(dx, dy, phi);
                matrix[0, 2] = df1dphi(dx, dy, phi);

                matrix[1, 0] = df2dx(dx, dy, phi);
                matrix[1, 1] = df2dy(dx, dy, phi);
                matrix[1, 2] = df2dphi(dx, dy, phi);

                matrix[2, 0] = df3dx(dx, dy, phi);
                matrix[2, 1] = df3dy(dx, dy, phi);
                matrix[2, 2] = df3dphi(dx, dy, phi);
                return matrix;
            }
            private double[] getF(double dx, double dy, double phi)
            {
                double[] result = new double[3];
                result[0] = -f1(dx, dy, phi);
                result[1] = -f2(dx, dy, phi);
                result[2] = -f3(dx, dy, phi);
                return result;
            }

            private double[] SolveNonLinearSystem(double e, int maxItarations = 10)
            {
                double[] xk = new double[3];
                int iteration = 0;
                while (iteration < maxItarations)
                {
                    double[] dxk = Matrix.SolveLinearSystem(getJacobian(xk[0], xk[1], xk[2]), getF(xk[0], xk[1], xk[2]));
                    if (dxk[2] * dxk[2] < e)
                        break;
                    else
                    {
                        xk[0] += dxk[0];
                        xk[1] += dxk[1];
                        xk[2] += dxk[2];
                    }
                    iteration++;
                }

                return xk;
            }
        }
    }
}
 