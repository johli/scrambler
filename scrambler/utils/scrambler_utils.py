import pandas as pd
import numpy as np
import scipy.sparse as sp

def get_sequence_masks(sequence_templates) :
    
    sequence_masks = [
        np.array([1 if sequence_templates[i][j] == '$' else 0 for j in range(len(sequence_templates[i]))])
        for i in range(len(sequence_templates))
    ]
    
    return sequence_masks

class BatchEncoder :
    
    def __init__(self, encoder, memory_efficient=False, memory_efficient_dump_size=30000) :
        self.encoder = encoder
        self.memory_efficient = memory_efficient
        self.memory_efficient_dump_size = memory_efficient_dump_size
    
    def encode(self, seqs) :
        
        batch_dims = tuple([len(seqs)] + list(self.encoder.encode_dims))
        encodings = np.zeros(batch_dims)
        
        self.encode_inplace(seqs, encodings)
        
        return encodings
    
    def encode_inplace(self, seqs, encodings) :
        for i in range(0, len(seqs)) :
            self.encoder.encode_inplace(seqs[i], encodings[i,])
    
    def encode_row_sparse(self, seqs) :
        return sp.csr_matrix(self.encode_sparse(seqs))
    
    def encode_col_sparse(self, seqs) :
        return sp.csc_matrix(self.encode_sparse(seqs))
    
    def encode_sparse(self, seqs) :
        n_cols = np.prod(np.ravel(list(self.encoder.encode_dims)))
        encoding_mat = None

        if not self.memory_efficient or len(seqs) <= self.memory_efficient_dump_size :
            encoding_mat = sp.lil_matrix((len(seqs), n_cols))
            for i in range(0, len(seqs)) :
                self.encoder.encode_inplace_sparse(seqs[i], encoding_mat, i)
        else :
            dump_counter = 0
            dump_max = self.memory_efficient_dump_size
            encoding_acc = None
            encoding_part = sp.lil_matrix((dump_max, n_cols))
            seqs_left = len(seqs)

            for i in range(0, len(seqs)) :
                if dump_counter >= dump_max :
                    if encoding_acc == None :
                        encoding_acc = sp.csr_matrix(encoding_part)
                    else :
                        encoding_acc = sp.vstack([encoding_acc, sp.csr_matrix(encoding_part)])
                    
                    if seqs_left >= dump_max :
                        encoding_part = sp.lil_matrix((dump_max, n_cols))
                    else :
                        encoding_part = sp.lil_matrix((seqs_left, n_cols))

                    dump_counter = 0
                
                dump_counter += 1
                seqs_left -= 1

                self.encoder.encode_inplace_sparse(seqs[i], encoding_part, i % dump_max)

            if encoding_part.shape[0] > 0 :
                encoding_acc = sp.vstack([encoding_acc, sp.csr_matrix(encoding_part)])

            encoding_mat = sp.csr_matrix(encoding_acc)
        
        return encoding_mat
    
    def decode(self, encodings) :
        decodings = []
        for i in range(0, encodings.shape[0]) :
            decodings.append(self.encoder.decode(encodings[i,]))
        
        return decodings
    
    def decode_sparse(self, encoding_mat) :
        decodings = []
        for i in range(0, encoding_mat.shape[0]) :
            decodings.append(self.encoder.decode_sparse(encoding_mat, i))
        
        return decodings
    
    def __call__(self, seqs) :
        return self.encode(seqs)

class SparseBatchEncoder(BatchEncoder) :
    
    def __init__(self, encoder, sparse_mode='row') :
        super(SparseBatchEncoder, self).__init__(encoder)
        
        self.sparse_mode = sparse_mode
    
    def encode(self, seqs) :
        return self.__call__(seqs)
    
    def decode(self, encodings) :
        return self.decode_sparse(encodings)
    
    def __call__(self, seqs) :
        if self.sparse_mode == 'row' :
            return self.encode_row_sparse(seqs)
        elif self.sparse_mode == 'col' :
            return self.encode_col_sparse(seqs)
        else :
            return self.encode_sparse(seqs)

class SequenceEncoder :
    
    def __init__(self, encoder_type_id, encode_dims) :
        self.encoder_type_id = encoder_type_id
        self.encode_dims = encode_dims
    
    def encode(self, seq) :
        raise NotImplementedError()
    
    def encode_inplace(self, seq, encoding) :
        raise NotImplementedError()
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        raise NotImplementedError()
    
    def decode(self, encoding) :
        raise NotImplementedError()
    
    def decode_sparse(self, encoding_mat, row_index) :
        raise NotImplementedError()
    
    def __call__(self, seq) :
        return self.encode(seq)
    
class OneHotEncoder(SequenceEncoder) :
    
    def __init__(self, seq_length, channel_map) :
        super(OneHotEncoder, self).__init__('onehot', (seq_length, len(channel_map)))
        
        self.seq_len = seq_length
        self.n_channels = len(channel_map)
        self.encode_map = channel_map
        self.decode_map = {
            val : key for key, val in channel_map.items()
        }
    
    def encode(self, seq) :
        encoding = np.zeros((self.seq_len, self.n_channels))
        
        for i in range(len(seq)) :
            if seq[i] in self.encode_map :
                channel_ix = self.encode_map[seq[i]]
                encoding[i, channel_ix] = 1.

        return encoding
    
    def encode_inplace(self, seq, encoding) :
        for i in range(len(seq)) :
            if seq[i] in self.encode_map :
                channel_ix = self.encode_map[seq[i]]
                encoding[i, channel_ix] = 1.
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        raise NotImplementError()
    
    def decode(self, encoding) :
        seq = ''
    
        for pos in range(0, encoding.shape[0]) :
            argmax_nt = np.argmax(encoding[pos, :])
            max_nt = np.max(encoding[pos, :])
            if max_nt == 1 :
                seq += self.decode_map[argmax_nt]
            else :
                seq += self.decode_map[self.n_channels - 1]

        return seq
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.array(encoding_mat[row_index, :].todense()).reshape(-1, 4)
        return self.decode(encoding)

class NMerEncoder(SequenceEncoder) :
    
    def __init__(self, n_mer_len=6, count_n_mers=True) :
        super(NMerEncoder, self).__init__('mer_' + str(n_mer_len), (4**n_mer_len, ))
        
        self.count_n_mers = count_n_mers
        self.n_mer_len = n_mer_len
        self.encode_order = ['A', 'C', 'G', 'T']
        self.n_mers = self._get_ordered_nmers(n_mer_len)
        
        self.encode_map = {
            n_mer : n_mer_index for n_mer_index, n_mer in enumerate(self.n_mers)
        }
        
        self.decode_map = {
            n_mer_index : n_mer for n_mer_index, n_mer in enumerate(self.n_mers)
        }
    
    def _get_ordered_nmers(self, n_mer_len) :
        
        if n_mer_len == 0 :
            return []
        
        if n_mer_len == 1 :
            return list(self.encode_order.copy())
        
        n_mers = []
        
        prev_n_mers = self._get_ordered_nmers(n_mer_len - 1)
        
        for _, prev_n_mer in enumerate(prev_n_mers) :
            for _, nt in enumerate(self.encode_order) :
                n_mers.append(prev_n_mer + nt)
        
        return n_mers
            
    def encode(self, seq) :
        n_mer_vec = np.zeros(self.n_mer_len)
        self.encode_inplace(seq, n_mer_vec)

        return n_mer_vec
    
    def encode_inplace(self, seq, encoding) :
        for i_start in range(0, len(seq) - self.n_mer_len + 1) :
            i_end = i_start + self.n_mer_len
            n_mer = seq[i_start:i_end]
            
            if n_mer in self.encode_map :
                if self.count_n_mers :
                    encoding[self.encode_map[n_mer]] += 1
                else :
                    encoding[self.encode_map[n_mer]] = 1
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        for i_start in range(0, len(seq) - self.n_mer_len + 1) :
            i_end = i_start + self.n_mer_len
            n_mer = seq[i_start:i_end]
            
            if n_mer in self.encode_map :
                if self.count_n_mers :
                    encoding_mat[row_index, self.encode_map[n_mer]] += 1
                else :
                    encoding_mat[row_index, self.encode_map[n_mer]] = 1
    
    def decode(self, encoding) :
        n_mers = {}
    
        for i in range(0, encoding.shape[0]) :
            if encoding[i] != 0 :
                n_mers[self.decode_map[i]] = encoding[i]

        return n_mers
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.ravel(encoding_mat[row_index, :].todense())
        return self.decode(encoding)
