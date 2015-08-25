using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace CUDAFingerprinting.Common.Serialization
{
    public class BinarySerializationHelper
    {
        private static BinaryFormatter _formatter = new BinaryFormatter();

        public static byte[] SerializeObject<T>(T toSerialize)
        {
            var ms = new MemoryStream();
            _formatter.Serialize(ms, toSerialize);
            return ms.ToArray();
        }

        public static T DeserializeObject<T>(byte[] toDeserialize)
        {
            var result = _formatter.Deserialize(new MemoryStream(toDeserialize));

            return (T) result;
        }

        public static void SaveSerializedValue(byte[] value, string path)
        {
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
            {
                using (BinaryWriter bw = new BinaryWriter(fs))
                {
                    bw.Write(value);
                }
            }
        }
    }
}
