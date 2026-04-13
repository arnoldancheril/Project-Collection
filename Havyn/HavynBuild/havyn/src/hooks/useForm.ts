import { useState, useCallback } from 'react';

interface FormErrors {
  [key: string]: string;
}

interface UseFormProps<T> {
  initialValues: T;
  validate?: (values: T) => FormErrors;
  onSubmit: (values: T) => void | Promise<void>;
}

export const useForm = <T extends { [key: string]: any }>({
  initialValues,
  validate,
  onSubmit,
}: UseFormProps<T>) => {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [touched, setTouched] = useState<{ [key: string]: boolean }>({});

  const handleChange = useCallback(
    (name: keyof T, value: any) => {
      setValues((prevValues) => ({
        ...prevValues,
        [name]: value,
      }));

      if (touched[name as string]) {
        if (validate) {
          const validationErrors = validate({
            ...values,
            [name]: value,
          });
          setErrors((prevErrors) => ({
            ...prevErrors,
            [name]: validationErrors[name as string] || '',
          }));
        }
      }
    },
    [touched, validate, values]
  );

  const handleBlur = useCallback(
    (name: keyof T) => {
      setTouched((prevTouched) => ({
        ...prevTouched,
        [name]: true,
      }));

      if (validate) {
        const validationErrors = validate(values);
        setErrors((prevErrors) => ({
          ...prevErrors,
          [name]: validationErrors[name as string] || '',
        }));
      }
    },
    [validate, values]
  );

  const handleSubmit = useCallback(async () => {
    if (validate) {
      const validationErrors = validate(values);
      setErrors(validationErrors);

      // Touch all fields
      const allTouched = Object.keys(values).reduce(
        (acc, key) => ({ ...acc, [key]: true }),
        {}
      );
      setTouched(allTouched);

      if (Object.keys(validationErrors).length === 0) {
        setIsSubmitting(true);
        try {
          await onSubmit(values);
        } catch (error) {
          console.error('Form submission error:', error);
        } finally {
          setIsSubmitting(false);
        }
      }
    } else {
      setIsSubmitting(true);
      try {
        await onSubmit(values);
      } catch (error) {
        console.error('Form submission error:', error);
      } finally {
        setIsSubmitting(false);
      }
    }
  }, [onSubmit, validate, values]);

  const resetForm = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  }, [initialValues]);

  return {
    values,
    errors,
    touched,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm,
  };
}; 