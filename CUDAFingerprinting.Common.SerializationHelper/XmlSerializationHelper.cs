using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;
using System.IO;

namespace CUDAFingerprinting.Common.SerializationHelper
{
    public class XmlSerializationHelper
    {
        public static string SerializeObject<T>(T toSerialize)
        {
            var xmlSerializer = new XmlSerializer(toSerialize.GetType());
            var textWriter = new StringWriter();

            xmlSerializer.Serialize(textWriter, toSerialize);
            return textWriter.ToString();
        }

        public static T DeserializeObject<T>(string toDeserialize)
        {
            var xmlSerializer = new XmlSerializer(typeof(T));

            var result = xmlSerializer.Deserialize(new StringReader(toDeserialize));
            return (T)result;
        }
    }
}
