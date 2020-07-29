import Dispatch
import TensorFlow

class C {
  private var tensor = Tensor(1)

  func updateTensor() {
    DispatchQueue.global().async {
      self.tensor += 1
    }

    print(self.tensor)
  }
}

let c = C()
c.updateTensor()
