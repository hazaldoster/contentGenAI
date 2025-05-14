-- This file contains your base schema definition
-- Migrations will be applied on top of this schema

-- Function to decrease credits based on content type and type
CREATE OR REPLACE FUNCTION public.decrease_credit()
RETURNS TRIGGER AS $$
DECLARE
    current_credit numeric;
    credit_deduction numeric := 1; -- Default deduction
BEGIN
    -- Determine credit deduction based on content_type and type
    IF NEW.content_type = 'creative-scene' THEN
        credit_deduction := 2; -- Deduct 2 credits for creative-scene
        RAISE NOTICE 'Content type is creative-scene, deducting 2 credits for generation ID: %', NEW.id;
    ELSIF (NEW.content_type = 'product-visual' AND NEW.type = 'image') THEN
        credit_deduction := 0.25; -- Deduct 0.25 credits for product-visual images
        RAISE NOTICE 'Content type is product-visual and type is image, deducting 0.25 credits for generation ID: %', NEW.id;
    ELSIF (NEW.content_type = 'video-image' AND NEW.type = 'image') THEN
        credit_deduction := 0.25; -- Deduct 0.25 credits for video-image images
        RAISE NOTICE 'Content type is video-image and type is image, deducting 0.25 credits for generation ID: %', NEW.id;
    ELSE
        RAISE NOTICE 'Using default credit deduction of 1 for generation ID: %', NEW.id;
    END IF;
    
    -- Update the credit in the same row that was just inserted
    UPDATE public.generations
    SET credit = credit - credit_deduction
    WHERE id = NEW.id
    RETURNING credit INTO current_credit; -- Get the updated credit value
    
    -- Log the credit decrease
    RAISE NOTICE 'Credit decreased for generation ID: %. Deduction: %. New credit: %', 
                 NEW.id, credit_deduction, current_credit;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create the trigger
DROP TRIGGER IF EXISTS after_generation_insert ON public.generations;
CREATE TRIGGER after_generation_insert
    AFTER INSERT ON public.generations
    FOR EACH ROW
    EXECUTE FUNCTION public.decrease_credit(); 