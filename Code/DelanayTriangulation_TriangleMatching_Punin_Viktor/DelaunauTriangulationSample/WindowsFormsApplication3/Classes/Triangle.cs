using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DelaunauTriangulationSample.Classes
{
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
    }
}
