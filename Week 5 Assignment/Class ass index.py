class Smartphone:
    def __init__(self, brand, model, storage, camera_mp):
        self.brand = brand
        self.model = model
        self.storage = storage  # in GB
        self.camera_mp = camera_mp  # in Megapixels
    def take_photo(self):
        return f"{self.model} took a photo with {self.camera_mp}MP camera 📸"
    def make_call(self, number):
        return f"{self.model} is calling {number} 📞"
    def __str__(self):
        return f"{self.brand} {self.model} with {self.storage}GB storage"
class SmartWatch(Smartphone):
    def __init__(self, brand, model, storage, camera_mp, strap_type):
        super().__init__(brand, model, storage, camera_mp)
        self.strap_type = strap_type
    def take_photo(self):
        return f"{self.model} took a low-res wrist photo 🤳"
    def track_heart_rate(self):
        return f"{self.model} is tracking heart rate ❤️‍🔥"
phone = Smartphone("Samsung", "Galaxy S22", 128, 50)
watch = SmartWatch("Apple", "Watch Series 7", 32, 2, "Silicone")
print(phone.take_photo())
print(watch.take_photo())  # Polymorphism in action
print(watch.track_heart_rate())
print(phone.make_call("+2547012345698"))
