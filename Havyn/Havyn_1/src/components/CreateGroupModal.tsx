import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Modal,
  TouchableOpacity,
  TouchableWithoutFeedback,
  Keyboard,
  ScrollView,
  Switch,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface CreateGroupModalProps {
  visible: boolean;
  onClose: () => void;
  onCreateGroup: (name: string, description: string, isPublic: boolean, tags: string[]) => void;
}

const CreateGroupModal: React.FC<CreateGroupModalProps> = ({
  visible,
  onClose,
  onCreateGroup,
}) => {
  const [groupName, setGroupName] = useState('');
  const [description, setDescription] = useState('');
  const [isPublic, setIsPublic] = useState(true);
  const [tagInput, setTagInput] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [errors, setErrors] = useState<{ name?: string; description?: string }>({});

  const handleCreateGroup = () => {
    // Validate inputs
    const newErrors: { name?: string; description?: string } = {};
    
    if (!groupName.trim()) {
      newErrors.name = 'Group name is required';
    }
    
    if (!description.trim()) {
      newErrors.description = 'Description is required';
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    // Create the group
    onCreateGroup(groupName, description, isPublic, tags);
    
    // Reset form
    setGroupName('');
    setDescription('');
    setIsPublic(true);
    setTags([]);
    setTagInput('');
    setErrors({});
    
    // Close modal
    onClose();
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !tags.includes(tagInput.trim())) {
      setTags([...tags, tagInput.trim()]);
      setTagInput('');
    }
  };

  const handleRemoveTag = (tag: string) => {
    setTags(tags.filter(t => t !== tag));
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={true}
      onRequestClose={onClose}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Create New Group</Text>
              <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                <Ionicons name="close" size={24} color="#666" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.formContainer}>
              <Text style={styles.label}>Group Name</Text>
              <TextInput
                style={[styles.input, errors.name && styles.inputError]}
                placeholder="e.g. Lincoln Park Roommates"
                value={groupName}
                onChangeText={setGroupName}
              />
              {errors.name && <Text style={styles.errorText}>{errors.name}</Text>}

              <Text style={styles.label}>Description</Text>
              <TextInput
                style={[styles.input, styles.textArea, errors.description && styles.inputError]}
                placeholder="What is this group about?"
                value={description}
                onChangeText={setDescription}
                multiline
                numberOfLines={4}
                textAlignVertical="top"
              />
              {errors.description && <Text style={styles.errorText}>{errors.description}</Text>}

              <View style={styles.switchContainer}>
                <Text style={styles.label}>Public Group</Text>
                <Switch
                  value={isPublic}
                  onValueChange={setIsPublic}
                  trackColor={{ false: '#ccc', true: '#3498db' }}
                  thumbColor={Platform.OS === 'ios' ? undefined : isPublic ? '#2980b9' : '#f4f3f4'}
                />
              </View>
              <Text style={styles.helperText}>
                {isPublic
                  ? 'Anyone can find and join this group'
                  : 'Only members can see this group'}
              </Text>

              <Text style={styles.label}>Tags</Text>
              <View style={styles.tagInputContainer}>
                <TextInput
                  style={styles.tagInput}
                  placeholder="Add tags (e.g. Downtown, Budget, Pet-friendly)"
                  value={tagInput}
                  onChangeText={setTagInput}
                  onSubmitEditing={handleAddTag}
                />
                <TouchableOpacity style={styles.addTagButton} onPress={handleAddTag}>
                  <Ionicons name="add" size={24} color="white" />
                </TouchableOpacity>
              </View>

              <View style={styles.tagsContainer}>
                {tags.map((tag, index) => (
                  <View key={index} style={styles.tag}>
                    <Text style={styles.tagText}>{tag}</Text>
                    <TouchableOpacity onPress={() => handleRemoveTag(tag)}>
                      <Ionicons name="close-circle" size={16} color="#666" />
                    </TouchableOpacity>
                  </View>
                ))}
              </View>
            </ScrollView>

            <View style={styles.buttonContainer}>
              <TouchableOpacity style={styles.cancelButton} onPress={onClose}>
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.createButton} onPress={handleCreateGroup}>
                <Text style={styles.createButtonText}>Create Group</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </TouchableWithoutFeedback>
    </Modal>
  );
};

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 16,
    width: '90%',
    maxHeight: '80%',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
    padding: 16,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  closeButton: {
    padding: 4,
  },
  formContainer: {
    padding: 16,
    maxHeight: '70%',
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
    marginTop: 16,
  },
  input: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    fontSize: 16,
  },
  inputError: {
    borderColor: '#e74c3c',
  },
  errorText: {
    color: '#e74c3c',
    fontSize: 14,
    marginTop: 4,
  },
  textArea: {
    minHeight: 100,
    paddingTop: 12,
  },
  switchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 16,
    marginBottom: 8,
  },
  helperText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  tagInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  tagInput: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    fontSize: 16,
  },
  addTagButton: {
    backgroundColor: '#3498db',
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 12,
    marginBottom: 16,
  },
  tag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    borderRadius: 16,
    paddingVertical: 6,
    paddingHorizontal: 12,
    marginRight: 8,
    marginBottom: 8,
  },
  tagText: {
    fontSize: 14,
    marginRight: 4,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  cancelButton: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    marginRight: 12,
    borderRadius: 8,
  },
  cancelButtonText: {
    fontSize: 16,
    color: '#666',
  },
  createButton: {
    backgroundColor: '#3498db',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  createButtonText: {
    fontSize: 16,
    color: 'white',
    fontWeight: '600',
  },
});

export default CreateGroupModal; 