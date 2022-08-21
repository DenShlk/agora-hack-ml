class Product:
    def __init__(self, product_id: str, name: str, props: [str], is_reference: bool, reference_id: str):
        self.product_id = product_id
        self.name = name
        self.props = props
        self.reference_id = reference_id
        self.is_reference = is_reference

    def __str__(self):
        return f"{'Reference' if self.is_reference else f'Referrer of {self.reference_id}'} {self.name} ({self.product_id}): \n {self.props}"