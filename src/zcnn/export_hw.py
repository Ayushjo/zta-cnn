"""Package the deployable model for the Phase II on-board test.

Writes artifacts/hw/:
  cifar10_zta.tflite           UINT8 model (copy of <DEPLOY_MODEL>_uint8.tflite)
  labels_cifar10.txt           one label per line (format of SW/fs/labels_*.txt)
  cifar10_input_<k>.bmp        32x32 24-bit BMP test images (read by TENSOR::CreateWithBitmap,
                               SW/base/tensor.cpp:89)
  cifar10_expected_<k>.bin     expected raw uint8 output tensor for that image, same format as
                               SW/fs/classifier.bin used by test_mobinet (SW/src/test.cpp:1029)
  golden.json                  labels, expected top-5 and TFLite outputs for all vectors
  test_cifar10.cpp.txt         drop-in on-board test modelled on test_mobinet
"""
import json
import shutil

import numpy as np
from PIL import Image

from . import config as C
from .data import load_cifar10
from .quantize import tflite_path
from .tflite_runner import TFLiteModel, ztachip_top5
from .ztachip_check import check

HW = C.ARTIFACTS / "hw"

TEST_CPP = r"""// On-board golden test for the CIFAR-10 model. Add to SW/src/test.cpp and call from
// the unit-test list. Model/label/BMP/BIN files must be reachable through fopen()
// (add them to SW/fs/bin2c.py + SW/base/newlib.c, or put them on the SD card).
void test_cifar10()
{
   static const char *bmp[] = { %(bmps)s };
   static const char *bin[] = { %(bins)s };
   for(int k=0;k < %(n)d;k++) {
      TENSOR input;
      TENSOR output;
      Graph graph;
      TfliteNn nn;
      ZtaStatus rc=input.CreateWithBitmap(bmp[k]);
      assert(rc==ZtaStatusOk);
      nn.Create("cifar10_zta.tflite",&input,1,&output);
      graph.Add(&nn);
      graph.Verify();
      FLUSH_DATA_CACHE();
      graph.Prepare();
      uint32_t t0=TimeGet();
      graph.RunUntilCompletion();
      uint32_t t1=TimeGet();
      FLUSH_DATA_CACHE();
      size_t size=output.GetBufLen();          // 10 bytes
      uint8_t expect[16];
      FILE *fp=fopen(bin[k],"rb");
      assert(fp && fread(expect,1,size,fp)==size);
      fclose(fp);
      uint8_t *got=(uint8_t *)output.GetBuf();
      int maxdiff=0;
      for(size_t i=0;i < size;i++) {
         int d=(int)got[i]-(int)expect[i];
         if(d<0) d=-d;
         if(d>maxdiff) maxdiff=d;
      }
      int top5[5];
      NeuralNet::GetTop5(got,size,top5);
      printf("cifar10 vector %%d: top1=%%d maxdiff=%%d time=%%d\n",k,top5[0],maxdiff,(int)(t1-t0));
      nn.Unload();
   }
}
"""


def main(n_vectors=10):
    HW.mkdir(exist_ok=True)
    src = tflite_path(C.DEPLOY_MODEL, "uint8")
    dst = HW / "cifar10_zta.tflite"
    shutil.copy(src, dst)
    (HW / "labels_cifar10.txt").write_text("\n".join(C.CLASS_NAMES) + "\n")
    c = check(dst, HW / "labels_cifar10.txt")
    assert c.passed, c.report()

    _, _, (x, y) = load_cifar10()
    m = TFLiteModel(dst)
    assert m.in_zp == 0 and abs(m.in_scale - 1 / 255) < 1e-6, "input must take raw bytes"
    # One correctly-classified image per class, deterministic.
    golden = dict(model=dst.name, source=src.name, vectors=[])
    bmps, bins = [], []
    for cls in range(n_vectors):
        for i in np.where(y == cls)[0]:
            raw = m.raw(x[i])
            if raw.argmax() == cls:
                break
        k = len(bins)
        Image.fromarray(x[i]).save(HW / f"cifar10_input_{k}.bmp")
        raw.astype(np.uint8).tofile(HW / f"cifar10_expected_{k}.bin")
        bmps.append(f'"cifar10_input_{k}.bmp"')
        bins.append(f'"cifar10_expected_{k}.bin"')
        golden["vectors"].append(dict(index=int(i), label=C.CLASS_NAMES[cls], raw_uint8=raw.tolist(),
                                      ztachip_top5=[(C.CLASS_NAMES[j], v, pct) for j, v, pct in ztachip_top5(raw)]))
    (HW / "golden.json").write_text(json.dumps(golden, indent=1))
    (HW / "test_cifar10.cpp.txt").write_text(TEST_CPP % dict(bmps=", ".join(bmps), bins=", ".join(bins), n=len(bins)))
    print(f"[export_hw] wrote {len(bins)} golden vectors + model + labels to {HW}")


if __name__ == "__main__":
    main()
