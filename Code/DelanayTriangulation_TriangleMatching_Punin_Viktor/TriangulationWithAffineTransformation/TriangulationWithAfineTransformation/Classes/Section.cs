using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TriangulationWithAfineTransformation.Classes
{

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

        public Triangle Left
        {
            get;
            set;
        }

        public Triangle Right
        {
            get;
            set;
        }

        public Section(Vector vector, Triangle left = null, Triangle right = null)
        {
            this.Vector = vector;
            this.Left = left;
            this.Right = right;
        }

        public static Section GetFrom(List<Section> sections, Vector vec)
        {
            foreach (Section section in sections)
            {
                if (section.Equals(vec))
                    return section;
            }
            return null;
        }

        public double CountAnglesSum(Point point)
        {
            return Vector.GetSumOfAngles(point);
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
}