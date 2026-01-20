# schemas/phone_spec.py
from pydantic import BaseModel
from typing import Optional

class PhoneSpec(BaseModel):
    brand: str
    model: str

    cpu_score: Optional[int]
    gpu_score: Optional[int]
    ram_gb: Optional[int]
    storage_type: Optional[str]

    display_type: Optional[str]
    refresh_rate: Optional[int]
    resolution_ppi: Optional[int]

    main_camera_mp: Optional[int]
    ois: Optional[bool]
    video_4k: Optional[bool]

    battery_mah: Optional[int]
    charging_watt: Optional[int]

    build_material: Optional[str]
    os_update_years: Optional[int]

    price_usd: Optional[float]
