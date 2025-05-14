-- Migration: add_credit_decrease_trigger
-- Description: Adds a trigger to decrease credit after a new generation is inserted

-- Updated function to decrease credits and log the action
CREATE OR REPLACE FUNCTION public.decrease_credit()
RETURNS TRIGGER AS $$
DECLARE
    current_credit integer;
BEGIN
    -- Update the credit in the same row that was just inserted
    UPDATE public.generations
    SET credit = credit - 1
    WHERE id = NEW.id
    RETURNING credit INTO current_credit; -- Get the updated credit value
    
    -- Log the credit decrease
    RAISE NOTICE 'Credit decreased for generation ID: %. New credit: %', NEW.id, current_credit;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Re-create the trigger (optional but good practice to ensure it uses the updated function)
DROP TRIGGER IF EXISTS after_generation_insert ON public.generations;
CREATE TRIGGER after_generation_insert
    AFTER INSERT ON public.generations
    FOR EACH ROW
    EXECUTE FUNCTION public.decrease_credit(); 