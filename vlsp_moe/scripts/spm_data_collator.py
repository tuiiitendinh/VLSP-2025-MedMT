import torch
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SPMDataCollatorForLanguageModeling:
    """
    Data collator cho mô hình ngôn ngữ SentencePiece.
    - Được thiết kế cho Causal Language Modeling (CLM).
    - Tự động tạo 'attention_mask' từ 'input_ids' và padding token.
    - Xử lý padding cho cả 'input_ids' và 'labels' một cách hiệu quả.
    """
    
    def __init__(self, tokenizer: Any, mlm: bool = False, pad_to_multiple_of: Optional[int] = None):
        """
        Args:
            tokenizer: Tokenizer được sử dụng (phải có thuộc tính 'pad_token_id' hoặc mặc định là 0).
            mlm (bool): Được giữ lại để tương thích, nhưng không được sử dụng cho Causal LM.
            pad_to_multiple_of (Optional[int]): Nếu được chỉ định, sẽ pad độ dài của batch đến bội số gần nhất của giá trị này.
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Hàm xử lý một batch các features từ Dataset.

        Args:
            features: Một list các dictionary, mỗi dictionary là một sample từ MoEDataset.

        Returns:
            Một dictionary chứa các tensor đã được pad sẵn sàng cho model.
        """
        # Lấy pad_token_id từ tokenizer. Mặc định là 0 nếu không được định nghĩa.
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)

        # Trích xuất các trường cần thiết từ list các features.
        # Đây là những trường được trả về bởi __getitem__ của MoEDataset.
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        expert_types = [f.get("expert_type", "general") for f in features]

        # Sử dụng hàm pad_sequence của PyTorch để padding hiệu quả.
        # Nó sẽ tự động pad theo độ dài của item dài nhất trong batch.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in input_ids_list],
            batch_first=True,
            padding_value=pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in labels_list],
            batch_first=True,
            padding_value=-100  # -100 là giá trị chuẩn để Pytorch bỏ qua khi tính loss.
        )

        # Thay vì cố gắng đọc 'attention_mask' từ features (gây ra KeyError),
        # chúng ta tạo ra nó bằng cách so sánh `input_ids` với `pad_token_id`.
        # Mask sẽ có giá trị 1 cho các token thực và 0 cho các token padding.
        attention_mask = (input_ids != pad_token_id).long()

        # Nếu pad_to_multiple_of được chỉ định, thực hiện padding thêm một lần nữa
        if self.pad_to_multiple_of is not None:
            max_length = input_ids.shape[1]
            new_max_length = ((max_length + self.pad_to_multiple_of - 1) 
                              // self.pad_to_multiple_of * self.pad_to_multiple_of)
            
            pad_length = new_max_length - max_length
            if pad_length > 0:
                # Pad cho input_ids
                input_ids = F.pad(input_ids, (0, pad_length), "constant", pad_token_id)
                # Pad cho attention_mask
                attention_mask = F.pad(attention_mask, (0, pad_length), "constant", 0)
                # Pad cho labels
                labels = F.pad(labels, (0, pad_length), "constant", -100)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # expert_type được trả về dưới dạng list, Trainer sẽ xử lý nó.
            "expert_type": expert_types,
        }
        
        return batch